/**
 * Context compaction for long sessions.
 *
 * Pure functions for compaction logic. The session manager handles I/O,
 * and after compaction the session is reloaded.
 */

import type { AgentMessage, StreamFn, ThinkingLevel } from "@earendil-works/pi-agent-core";
import { contentText, type RetryCallbacks, type RetryPolicy, retryAssistantCall, uuidv7 } from "@earendil-works/pi-ai";
import type { AssistantMessage, Context, Model, SimpleStreamOptions, Usage } from "@earendil-works/pi-ai/compat";
import { completeSimple } from "@earendil-works/pi-ai/compat";
import chalk from "chalk";
import { convertToLlm } from "../messages.ts";
import {
	buildSessionContext,
	type CompactionEntry,
	type SessionEntry,
	sessionEntryToContextMessages,
} from "../session-manager.ts";
import {
	computeFileLists,
	createFileOps,
	extractFileOpsFromMessage,
	type FileOperations,
	formatFileOperations,
	SUMMARIZATION_SYSTEM_PROMPT,
	serializeConversation,
} from "./utils.ts";

// ============================================================================
// File Operation Tracking
// ============================================================================

/** Details stored in CompactionEntry.details for file tracking */
export interface CompactionDetails {
	readFiles: string[];
	modifiedFiles: string[];
}

/**
 * Extract file operations from messages and previous compaction entries.
 */
function extractFileOperations(
	messages: AgentMessage[],
	entries: SessionEntry[],
	prevCompactionIndex: number,
): FileOperations {
	const fileOps = createFileOps();

	// Collect from previous compaction's details (if pi-generated)
	if (prevCompactionIndex >= 0) {
		const prevCompaction = entries[prevCompactionIndex] as CompactionEntry;
		if (!prevCompaction.fromHook && prevCompaction.details) {
			// fromHook field kept for session file compatibility
			const details = prevCompaction.details as CompactionDetails;
			if (Array.isArray(details.readFiles)) {
				for (const f of details.readFiles) fileOps.read.add(f);
			}
			if (Array.isArray(details.modifiedFiles)) {
				for (const f of details.modifiedFiles) fileOps.edited.add(f);
			}
		}
	}

	// Extract from tool calls in messages
	for (const msg of messages) {
		extractFileOpsFromMessage(msg, fileOps);
	}

	return fileOps;
}

// ============================================================================
// Message Extraction
// ============================================================================

/**
 * Extract AgentMessage from an entry if it produces one.
 * Returns undefined for entries that don't contribute to LLM context.
 */
function getMessageFromEntryForCompaction(entry: SessionEntry): AgentMessage | undefined {
	if (entry.type === "compaction") {
		return undefined;
	}
	return sessionEntryToContextMessages(entry)[0];
}

/** Result from compact() - SessionManager adds uuid/parentUuid when saving */
export interface CompactionResult<T = unknown> {
	summary: string;
	firstKeptEntryId: string;
	tokensBefore: number;
	estimatedTokensAfter?: number;
	/** Usage from the LLM call(s) that generated this summary, if available */
	usage?: Usage;
	/** Extension-specific data (e.g., ArtifactIndex, version markers for structured compaction) */
	details?: T;
}

function combineUsage(first: Usage, second: Usage): Usage {
	return {
		input: first.input + second.input,
		output: first.output + second.output,
		cacheRead: first.cacheRead + second.cacheRead,
		cacheWrite: first.cacheWrite + second.cacheWrite,
		...(first.cacheWrite1h !== undefined || second.cacheWrite1h !== undefined
			? { cacheWrite1h: (first.cacheWrite1h ?? 0) + (second.cacheWrite1h ?? 0) }
			: {}),
		...(first.reasoning !== undefined || second.reasoning !== undefined
			? { reasoning: (first.reasoning ?? 0) + (second.reasoning ?? 0) }
			: {}),
		totalTokens: first.totalTokens + second.totalTokens,
		cost: {
			input: first.cost.input + second.cost.input,
			output: first.cost.output + second.cost.output,
			cacheRead: first.cost.cacheRead + second.cost.cacheRead,
			cacheWrite: first.cost.cacheWrite + second.cost.cacheWrite,
			total: first.cost.total + second.cost.total,
		},
	};
}

// ============================================================================
// Types
// ============================================================================

export interface CompactionSettings {
	enabled: boolean;
	reserveTokens: number;
	keepRecentTokens: number;
}

export const DEFAULT_COMPACTION_SETTINGS: CompactionSettings = {
	enabled: true,
	reserveTokens: 16384,
	keepRecentTokens: 20000,
};

// ============================================================================
// Token calculation
// ============================================================================

/**
 * Calculate total context tokens from usage.
 * Uses the native totalTokens field when available, falls back to computing from components.
 */
export function calculateContextTokens(usage: Usage): number {
	return usage.totalTokens || usage.input + usage.output + usage.cacheRead + usage.cacheWrite;
}

const warnedContextWindows = new Set<number>();

/**
 * Multiplicative ratio beyond which a provider-reported context-token figure is treated as a
 * reporting anomaly relative to its own session's real, independently-measured content
 * (`sumMessageTokens` over the branch's messages) - even when the reported figure is
 * comfortably under the model's context window (see `MIN_REPORTED_TOKENS_FOR_RATIO_CHECK`
 * below for why the ratio alone is not sufficient). `estimateTokens`' chars/4 heuristic is
 * documented as conservative (it overestimates real tokenizer counts), so ordinary drift runs
 * the other way - measured content at or above the real reported figure, not far below it. A
 * reported figure legitimately several times its own session's measured content essentially
 * never happens in steady state. Both production incidents this guards against overstated the
 * real content by close to an order of magnitude or more (819,151 vs ~85,600 measured, ~9.6x;
 * 1,782,723 vs ~75,000 measured, ~23.8x - the latter already caught by the window check below,
 * but both clear this ratio with room to spare), so 5x leaves a comfortable margin under either
 * incident while staying well above the <1x drift a healthy session should show.
 */
const REPORTED_VS_MEASURED_IMPLAUSIBILITY_RATIO = 5;

/**
 * Floor on the reported figure itself (not the measured one) below which the ratio check above
 * is skipped outright. `usage.totalTokens` reflects the full request sent to the provider -
 * system prompt and tool definitions included - while `sumMessageTokens` only estimates the
 * conversation's own messages; neither the system prompt nor tool definitions are part of the
 * `messages` array this module measures. That fixed overhead is entirely legitimate and can
 * dominate the ratio on an early, short session (a large tool corpus outweighing a two-message
 * conversation), which is exactly the false-positive shape "ordinary, expected drift" has to
 * survive. Gating on the reported figure's own absolute size bounds this: below the floor, even
 * a wildly wrong ratio cannot meaningfully move `shouldCompact()` against a real model context
 * window, so there is nothing worth flagging.
 */
const MIN_REPORTED_TOKENS_FOR_RATIO_CHECK = 50_000;

/**
 * A provider-reported context-token figure is untrustworthy in either of two ways: it can
 * exceed the model's own context window outright (physically impossible - the API would have
 * refused a request that large), or it can sit comfortably under the window yet still wildly
 * overstate the session's real, independently-measured content (see
 * `REPORTED_VS_MEASURED_IMPLAUSIBILITY_RATIO` above - observed in production as a
 * `claude-code-cli` usage.totalTokens reading ~9.6x its session's real content while still
 * comfortably under a 1,000,000-token window, so the window check alone let it through). The
 * above-window check is a special case of the same underlying question - "is this figure
 * consistent with what this session could actually contain" - rather than an independent guard,
 * so both live in one function with one call site per caller.
 */
function isImplausibleContextTokens(reportedTokens: number, measuredTokens: number, contextWindow: number): boolean {
	if (contextWindow <= 0) return false;
	if (reportedTokens > contextWindow) return true;
	return exceedsPlausibleRatio(reportedTokens, measuredTokens);
}

/**
 * The reported-vs-baseline ratio dimension of the plausibility question, shared by both callers
 * so the threshold semantics live in exactly one place. `baselineTokens` is whatever independent
 * figure the caller can hold the reported one up against: a message-size sum for
 * {@link resolveReportedContextTokens}, the session's own previous reported usage for
 * {@link resolveOverflowPlausibility}. Both constants above document why a ratio alone is not
 * enough and why the absolute floor applies to the reported figure rather than the baseline.
 */
function exceedsPlausibleRatio(reportedTokens: number, baselineTokens: number): boolean {
	return (
		reportedTokens > MIN_REPORTED_TOKENS_FOR_RATIO_CHECK &&
		reportedTokens > baselineTokens * REPORTED_VS_MEASURED_IMPLAUSIBILITY_RATIO
	);
}

/**
 * Format the one-shot implausible-context-tokens warning message. This module has no notion
 * of whether a live UI currently owns the terminal (a raw stdout write would corrupt a TUI's
 * managed alternate-screen render - see `ExtensionRunner.hasUI()` and its existing gated
 * `console.warn` at extensions/runner.ts:551 for the established precedent), so formatting is
 * kept separate from printing: callers that know their own UI context render this string
 * through `console.warn()` themselves, only when appropriate.
 */
export function formatImplausibleContextTokensWarning(
	reportedTokens: number,
	measuredTokens: number,
	contextWindow: number,
): string {
	return chalk.yellow(
		`Warning: provider-reported context usage (${reportedTokens.toLocaleString()} tokens) is not plausible ` +
			`for this session (real measured content: ${measuredTokens.toLocaleString()} tokens, model context ` +
			`window: ${contextWindow.toLocaleString()} tokens) and was rejected as implausible; using the ` +
			`message-size estimate instead.`,
	);
}

/**
 * One-shot gate for surfacing a rejected provider usage figure, per distinct context window
 * per process, mirroring the dedup pattern used for deprecation warnings (see
 * utils/deprecation.ts). A still-affected provider stream tends to keep reporting anomalous
 * readings on every subsequent turn (per the production incident this guards against), so
 * deduping per-window rather than per-exact-value avoids re-warning every turn while still
 * surfacing the anomaly. Consuming the gate (returning true) does not depend on whether the
 * caller actually goes on to print anything.
 */
function shouldAnnounceImplausibleContextTokens(contextWindow: number): boolean {
	if (warnedContextWindows.has(contextWindow)) return false;
	warnedContextWindows.add(contextWindow);
	return true;
}

/** Clear one-shot implausible-context-tokens warning state. Exported for tests. */
export function clearContextTokensWarningsForTests(): void {
	warnedContextWindows.clear();
}

function sumMessageTokens(messages: AgentMessage[]): number {
	let estimated = 0;
	for (const message of messages) {
		estimated += estimateTokens(message);
	}
	return estimated;
}

/**
 * Vet a single provider-reported context-token figure against the model's context window.
 *
 * Callers that already hold the exact reported figure they intend to act on must use this
 * rather than re-deriving one through {@link estimateContextTokens}: that function picks the
 * *last valid* usage in the array and deliberately skips aborted and errored messages, so it
 * can end up judging the plausibility of a different message than the one the caller's figure
 * came from - or of no message at all.
 *
 * @returns The reported figure when it is plausible, or a message-size sum over `messages`
 * when it is not. `warning` carries the one-shot anomaly message (only on the first rejection
 * for this context window) for the caller to print through its own UI-aware channel - this
 * function never writes to the console itself.
 */
export function resolveReportedContextTokens(
	reportedTokens: number,
	messages: AgentMessage[],
	contextWindow: number,
): { tokens: number; rejected: boolean; warning?: string } {
	if (contextWindow <= 0) {
		return { tokens: reportedTokens, rejected: false };
	}
	const measuredTokens = sumMessageTokens(messages);
	if (!isImplausibleContextTokens(reportedTokens, measuredTokens, contextWindow)) {
		return { tokens: reportedTokens, rejected: false };
	}
	const warning = shouldAnnounceImplausibleContextTokens(contextWindow)
		? formatImplausibleContextTokensWarning(reportedTokens, measuredTokens, contextWindow)
		: undefined;
	return { tokens: measuredTokens, rejected: true, warning };
}

/**
 * Format the one-shot warning for a provider-reported *overflow* signal that was rejected.
 * Kept separate from {@link formatImplausibleContextTokensWarning} because the two rejections
 * have different consequences: that one substitutes a message-size estimate for the figure,
 * while this one discards an overflow classification and leaves the turn alone. Printing is
 * the caller's job for the same UI-ownership reason documented there.
 */
function formatImplausibleOverflowWarning(
	reportedTokens: number,
	baselineTokens: number,
	contextWindow: number,
): string {
	return chalk.yellow(
		`Warning: provider-reported context usage (${reportedTokens.toLocaleString()} tokens) is not plausible ` +
			`for this session (session baseline: ${baselineTokens.toLocaleString()} tokens, model context ` +
			`window: ${contextWindow.toLocaleString()} tokens) and was rejected as implausible; the reported ` +
			`context overflow was ignored.`,
	);
}

/**
 * Find the session's own last valid reported context size *before* `judgedMessage`, so the
 * message under judgement cannot serve as its own baseline. Uses the same last-valid-usage walk
 * (and the same aborted/errored/all-zero skipping) `getLastAssistantUsageInfo` performs.
 *
 * `compactionBoundaryTimestamp` (epoch ms of the latest compaction entry, when the session has
 * one) excludes candidates from before that boundary, mirroring the identical stale-usage check
 * the threshold-compaction path already performs. A compaction keeps its most recent messages
 * verbatim - usage field included - while dropping everything it summarized, so a kept assistant
 * still reports the old, far larger context. Left in play it would hand this guard an inflated
 * baseline that waves through exactly the corrupted readings the guard exists to reject.
 *
 * `hadPriorAssistantTurn` reports whether the walk saw *any* earlier assistant message, usable
 * or not. Its callers must not read a missing `tokens` as "this is the session's first turn":
 * the walk skips aborted, errored and all-zero-usage messages, so a long session that hit
 * persistent API errors yields no figure while `messages` has nonetheless accumulated a full
 * conversation worth measuring.
 */
function findPriorReportedContextTokens(
	messages: AgentMessage[],
	judgedMessage: AgentMessage,
	compactionBoundaryTimestamp?: number,
): { tokens?: number; hadPriorAssistantTurn: boolean } {
	const judgedIndex = messages.lastIndexOf(judgedMessage);
	const start = judgedIndex >= 0 ? judgedIndex - 1 : messages.length - 1;
	let hadPriorAssistantTurn = false;
	for (let i = start; i >= 0; i--) {
		const message = messages[i];
		if (message.role !== "assistant") continue;
		hadPriorAssistantTurn = true;
		const usage = getAssistantUsage(message);
		if (!usage) continue;
		if (
			compactionBoundaryTimestamp !== undefined &&
			(message as AssistantMessage).timestamp <= compactionBoundaryTimestamp
		) {
			return { hadPriorAssistantTurn };
		}
		return { tokens: calculateContextTokens(usage), hadPriorAssistantTurn };
	}
	return { hadPriorAssistantTurn };
}

/**
 * Vet `isContextOverflow`'s z.ai-style silent-overflow signal - a successful response whose
 * usage.input + usage.cacheRead exceeds contextWindow - against the session's own previous
 * reported usage.
 *
 * Unlike {@link resolveReportedContextTokens}, the reported figure sitting at or above
 * `contextWindow` is not itself evidence of anomaly here: that is exactly the condition this
 * overflow case is built to detect (a provider that accepts a request larger than its documented
 * window - see overflow.ts). Only the ratio dimension applies, and it prefers the session's own
 * previous reported usage over a bare `sumMessageTokens`: that sum measures the `messages` array
 * alone, excluding the system prompt and tool definitions, which is precisely the overhead that
 * dominates in the overflow regime - a tool-heavy session can legitimately report several times
 * its own measured message content without anything being corrupted. A previous reported figure
 * carries that same roughly-constant overhead, so the comparison stays apples-to-apples and only
 * a reading that jumps far beyond what this very session reported one turn ago (the failure mode
 * Case 3's guard defends against) is rejected.
 *
 * That previous figure is only usable as a baseline while it is itself credible. A provider stuck
 * in an anomalous reporting mode keeps overstating on every turn, so an unvetted predecessor
 * would let the second such reading validate itself against the first and defeat the guard from
 * then on. The predecessor therefore faces the full {@link isImplausibleContextTokens} test -
 * both the ratio against `sumMessageTokens` and the context window - before it is trusted. The
 * window dimension is waived for the judged reading only because an above-window figure is the
 * very signal under judgement; a *predecessor* enjoys no such exemption, because a predecessor
 * whose own above-window reading had been believed would already have forced a compaction on its
 * own turn and be excluded here by `compactionBoundaryTimestamp`. One that survives to this walk
 * is therefore a reading this guard already refused, and it must not come back a turn later as
 * the authority that waves its successor through. For the ratio dimension, since
 * `compactionBoundaryTimestamp` confines candidates to the current post-compaction segment,
 * which only ever grows, the current message-size sum is an upper bound on the smaller content
 * that existed at that earlier turn and no historical snapshot is needed. A predecessor that
 * fails either dimension is discarded in favour of the measured sum alone. A credible one is
 * combined with the measured sum via `max`, so a genuinely large, self-consistent history still
 * gets the higher and more accurate baseline.
 *
 * The ways that leaves no prior reading are not equivalent. On a session's very first turn
 * `messages` holds little more than the user's prompt, so it measures almost nothing about the
 * request the provider actually sized and cannot serve as a baseline at all - the reading is
 * trusted, matching the no-baseline-available behavior of {@link resolveReportedContextTokens}
 * for an unknown context window. That concession is confined to a genuinely first turn, i.e. one
 * with no earlier assistant message whatsoever: once the session has taken turns whose usage was
 * merely unusable (aborted, errored, all-zero), `messages` holds a real conversation and its
 * measured sum is a baseline, so the guard still applies. On the first turn after a compaction
 * the array likewise holds a complete picture of the current context content (the summary plus
 * the kept tail), so the measured sum stands alone as the baseline, the same fallback an
 * untrustworthy predecessor gets.
 *
 * @returns `plausible: false` when the reported figure should be rejected as an overflow signal
 * (the caller should not treat this as a real overflow). `warning` carries the one-shot anomaly
 * message (see `shouldAnnounceImplausibleContextTokens`) for the caller to print through its own
 * UI-aware channel when the reading is rejected.
 */
export function resolveOverflowPlausibility(
	reportedTokens: number,
	messages: AgentMessage[],
	judgedMessage: AgentMessage,
	contextWindow: number,
	compactionBoundaryTimestamp?: number,
): { plausible: boolean; warning?: string } {
	if (contextWindow <= 0) {
		return { plausible: true };
	}
	const prior = findPriorReportedContextTokens(messages, judgedMessage, compactionBoundaryTimestamp);
	if (!prior.hadPriorAssistantTurn && compactionBoundaryTimestamp === undefined) {
		return { plausible: true };
	}
	const measuredTokens = sumMessageTokens(messages);
	const priorReportedTokens = prior.tokens;
	const baselineTokens =
		priorReportedTokens === undefined ||
		isImplausibleContextTokens(priorReportedTokens, measuredTokens, contextWindow)
			? measuredTokens
			: Math.max(priorReportedTokens, measuredTokens);
	if (!exceedsPlausibleRatio(reportedTokens, baselineTokens)) {
		return { plausible: true };
	}
	const warning = shouldAnnounceImplausibleContextTokens(contextWindow)
		? formatImplausibleOverflowWarning(reportedTokens, baselineTokens, contextWindow)
		: undefined;
	return { plausible: false, warning };
}

/**
 * Get usage from an assistant message if available.
 * Skips aborted, error, and all-zero usage messages as they don't have valid usage data.
 */
function getAssistantUsage(msg: AgentMessage): Usage | undefined {
	if (msg.role === "assistant" && "usage" in msg) {
		const assistantMsg = msg as AssistantMessage;
		if (
			assistantMsg.stopReason !== "aborted" &&
			assistantMsg.stopReason !== "error" &&
			assistantMsg.usage &&
			calculateContextTokens(assistantMsg.usage) > 0
		) {
			return assistantMsg.usage;
		}
	}
	return undefined;
}

/**
 * Find the last valid assistant message usage from session entries.
 */
export function getLastAssistantUsage(entries: SessionEntry[]): Usage | undefined {
	for (let i = entries.length - 1; i >= 0; i--) {
		const entry = entries[i];
		if (entry.type === "message") {
			const usage = getAssistantUsage(entry.message);
			if (usage) return usage;
		}
	}
	return undefined;
}

export interface ContextUsageEstimate {
	tokens: number;
	usageTokens: number;
	trailingTokens: number;
	lastUsageIndex: number | null;
	/**
	 * True when the reported usage was rejected as implausible - either it exceeded the model's
	 * context window outright, or it deviated too far from the session's real, measured content
	 * (see `isImplausibleContextTokens` in compaction.ts).
	 */
	usageRejected?: boolean;
	/** One-shot anomaly message when `usageRejected` is true and this is the first such rejection for this context window - print through the caller's own UI-aware channel. */
	warning?: string;
}

function getLastAssistantUsageInfo(messages: AgentMessage[]): { usage: Usage; index: number } | undefined {
	for (let i = messages.length - 1; i >= 0; i--) {
		const usage = getAssistantUsage(messages[i]);
		if (usage) return { usage, index: i };
	}
	return undefined;
}

/**
 * Estimate context tokens from messages, using the last assistant usage when available.
 * If there are messages after the last usage, estimate their tokens with estimateTokens.
 *
 * @param contextWindow When provided, a reported usage figure that exceeds this window, or
 * that deviates too far from the session's own real, measured content, is rejected as a
 * provider reporting anomaly rather than trusted as the real context size; the estimate falls
 * back to a message-size sum over every message instead (see `isImplausibleContextTokens`).
 * Omit when the model's context window is not known to the caller (the reported figure is
 * then trusted as before - callers that know the window should always pass it).
 */
export function estimateContextTokens(messages: AgentMessage[], contextWindow?: number): ContextUsageEstimate {
	const usageInfo = getLastAssistantUsageInfo(messages);

	if (!usageInfo) {
		const estimated = sumMessageTokens(messages);
		return {
			tokens: estimated,
			usageTokens: 0,
			trailingTokens: estimated,
			lastUsageIndex: null,
		};
	}

	const reportedUsageTokens = calculateContextTokens(usageInfo.usage);

	if (contextWindow !== undefined) {
		const resolved = resolveReportedContextTokens(reportedUsageTokens, messages, contextWindow);
		if (resolved.rejected) {
			return {
				tokens: resolved.tokens,
				usageTokens: 0,
				trailingTokens: resolved.tokens,
				lastUsageIndex: usageInfo.index,
				usageRejected: true,
				warning: resolved.warning,
			};
		}
	}

	let trailingTokens = 0;
	for (let i = usageInfo.index + 1; i < messages.length; i++) {
		trailingTokens += estimateTokens(messages[i]);
	}

	return {
		tokens: reportedUsageTokens + trailingTokens,
		usageTokens: reportedUsageTokens,
		trailingTokens,
		lastUsageIndex: usageInfo.index,
	};
}

/**
 * Check if compaction should trigger based on context usage.
 */
export function shouldCompact(contextTokens: number, contextWindow: number, settings: CompactionSettings): boolean {
	if (!settings.enabled) return false;
	return contextTokens > contextWindow - settings.reserveTokens;
}

// ============================================================================
// Cut point detection
// ============================================================================

const ESTIMATED_IMAGE_CHARS = 4800;

function estimateTextAndImageContentChars(content: string | Array<{ type: string; text?: string }>): number {
	if (typeof content === "string") {
		return content.length;
	}

	let chars = 0;
	for (const block of content) {
		if (block.type === "text" && block.text) {
			chars += block.text.length;
		} else if (block.type === "image") {
			chars += ESTIMATED_IMAGE_CHARS;
		}
	}
	return chars;
}

/**
 * Estimate token count for a message using chars/4 heuristic.
 * This is conservative (overestimates tokens).
 */
export function estimateTokens(message: AgentMessage): number {
	let chars = 0;

	switch (message.role) {
		case "user": {
			chars = estimateTextAndImageContentChars(
				(message as { content: string | Array<{ type: string; text?: string }> }).content,
			);
			return Math.ceil(chars / 4);
		}
		case "assistant": {
			const assistant = message as AssistantMessage;
			for (const block of assistant.content) {
				if (block.type === "text") {
					chars += block.text.length;
				} else if (block.type === "thinking") {
					chars += block.thinking.length;
				} else if (block.type === "toolCall") {
					chars += block.name.length + JSON.stringify(block.arguments).length;
				}
			}
			return Math.ceil(chars / 4);
		}
		case "custom":
		case "toolResult": {
			chars = estimateTextAndImageContentChars(message.content);
			return Math.ceil(chars / 4);
		}
		case "bashExecution": {
			chars = message.command.length + message.output.length;
			return Math.ceil(chars / 4);
		}
		case "branchSummary":
		case "compactionSummary": {
			chars = message.summary.length;
			return Math.ceil(chars / 4);
		}
	}

	return 0;
}

function isCutPointMessage(message: AgentMessage): boolean {
	switch (message.role) {
		case "user":
		case "assistant":
		case "bashExecution":
		case "custom":
		case "branchSummary":
		case "compactionSummary":
			return true;
		case "toolResult":
			return false;
	}
	return false;
}

function isTurnStartMessage(message: AgentMessage): boolean {
	switch (message.role) {
		case "user":
		case "bashExecution":
		case "custom":
		case "branchSummary":
		case "compactionSummary":
			return true;
		case "assistant":
		case "toolResult":
			return false;
	}
	return false;
}

function isTurnStartEntry(entry: SessionEntry): boolean {
	if (entry.type === "compaction") {
		return false;
	}
	return sessionEntryToContextMessages(entry).some(isTurnStartMessage);
}

/**
 * Find valid cut points: indices of context-visible user-like or assistant messages.
 * Never cut at tool results (they must follow their tool call).
 * When we cut at an assistant message with tool calls, its tool results follow it
 * and will be kept.
 */
function findValidCutPoints(entries: SessionEntry[], startIndex: number, endIndex: number): number[] {
	const cutPoints: number[] = [];
	for (let i = startIndex; i < endIndex; i++) {
		const entry = entries[i];
		if (entry.type === "compaction") {
			continue;
		}
		if (sessionEntryToContextMessages(entry).some(isCutPointMessage)) {
			cutPoints.push(i);
		}
	}
	return cutPoints;
}

/**
 * Find the context-visible user-role message that starts the turn containing the given entry index.
 * Returns -1 if no turn start found before the index.
 */
export function findTurnStartIndex(entries: SessionEntry[], entryIndex: number, startIndex: number): number {
	for (let i = entryIndex; i >= startIndex; i--) {
		if (isTurnStartEntry(entries[i])) {
			return i;
		}
	}
	return -1;
}

export interface CutPointResult {
	/** Index of first entry to keep */
	firstKeptEntryIndex: number;
	/** Index of user message that starts the turn being split, or -1 if not splitting */
	turnStartIndex: number;
	/** Whether this cut splits a turn (cut point is not a user message) */
	isSplitTurn: boolean;
}

/**
 * Find the cut point in session entries that keeps approximately `keepRecentTokens`.
 *
 * Algorithm: Walk backwards from newest, accumulating estimated message sizes.
 * Stop when we've accumulated >= keepRecentTokens. Cut at that point.
 *
 * Can cut at user OR assistant messages (never tool results). When cutting at an
 * assistant message with tool calls, its tool results come after and will be kept.
 *
 * Returns CutPointResult with:
 * - firstKeptEntryIndex: the entry index to start keeping from
 * - turnStartIndex: if cutting mid-turn, the user message that started that turn
 * - isSplitTurn: whether we're cutting in the middle of a turn
 *
 * Only considers entries between `startIndex` and `endIndex` (exclusive).
 */
export function findCutPoint(
	entries: SessionEntry[],
	startIndex: number,
	endIndex: number,
	keepRecentTokens: number,
): CutPointResult {
	const cutPoints = findValidCutPoints(entries, startIndex, endIndex);

	if (cutPoints.length === 0) {
		return { firstKeptEntryIndex: startIndex, turnStartIndex: -1, isSplitTurn: false };
	}

	// Walk backwards from newest, accumulating estimated message sizes
	let accumulatedTokens = 0;
	let cutIndex = cutPoints[0]; // Default: keep from first message (not header)

	for (let i = endIndex - 1; i >= startIndex; i--) {
		const entry = entries[i];
		const messageTokens = sessionEntryToContextMessages(entry).reduce(
			(sum, message) => sum + estimateTokens(message),
			0,
		);
		if (messageTokens === 0) continue;
		accumulatedTokens += messageTokens;

		// Check if we've exceeded the budget
		if (accumulatedTokens >= keepRecentTokens) {
			// Find the closest valid cut point at or after this entry
			for (let c = 0; c < cutPoints.length; c++) {
				if (cutPoints[c] >= i) {
					cutIndex = cutPoints[c];
					break;
				}
			}
			break;
		}
	}

	// Scan backwards from cutIndex to include adjacent metadata entries that do not affect context.
	while (cutIndex > startIndex) {
		const prevEntry = entries[cutIndex - 1];
		// Stop at compaction boundaries or context-visible entries.
		if (prevEntry.type === "compaction" || sessionEntryToContextMessages(prevEntry).length > 0) {
			break;
		}
		cutIndex--;
	}

	// Determine if this is a split turn
	const cutEntry = entries[cutIndex];
	const startsTurn = isTurnStartEntry(cutEntry);
	const turnStartIndex = startsTurn ? -1 : findTurnStartIndex(entries, cutIndex, startIndex);

	return {
		firstKeptEntryIndex: cutIndex,
		turnStartIndex,
		isSplitTurn: !startsTurn && turnStartIndex !== -1,
	};
}

// ============================================================================
// Summarization
// ============================================================================

const SUMMARIZATION_PROMPT = `The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work.

Use this EXACT format:

## Goal
[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by user]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, examples, or references needed to continue]
- [Or "(none)" if not applicable]

Keep each section concise. Preserve exact file paths, function names, and error messages.`;

const UPDATE_SUMMARIZATION_INSTRUCTIONS = `Update the existing structured summary with new information. RULES:
- PRESERVE all existing information from the previous summary
- ADD new progress, decisions, and context from the new messages
- UPDATE the Progress section: move items from "In Progress" to "Done" when completed
- UPDATE "Next Steps" based on what was accomplished
- PRESERVE exact file paths, function names, and error messages
- If something is no longer relevant, you may remove it

Use this EXACT format:

## Goal
[Preserve existing goals, add new ones if the task expanded]

## Constraints & Preferences
- [Preserve existing, add new ones discovered]

## Progress
### Done
- [x] [Include previously done items AND newly completed items]

### In Progress
- [ ] [Current work - update based on progress]

### Blocked
- [Current blockers - remove if resolved]

## Key Decisions
- **[Decision]**: [Brief rationale] (preserve all previous, add new)

## Next Steps
1. [Update based on current state]

## Critical Context
- [Preserve important context, add new if needed]

Keep each section concise. Preserve exact file paths, function names, and error messages.`;

const UPDATE_SUMMARIZATION_PROMPT = `The messages above are NEW conversation messages to incorporate into the existing summary provided in <previous-summary> tags.

${UPDATE_SUMMARIZATION_INSTRUCTIONS}`;

/**
 * Returns an error message when a summarization response cannot safely be persisted.
 * A length stop contains partial text and must not become a session checkpoint.
 */
export function getSummarizationFailure(response: AssistantMessage, label: string): string | undefined {
	if (response.stopReason === "error") {
		return `${label} failed: ${response.errorMessage || "Unknown error"}`;
	}
	if (response.stopReason === "length") {
		return `${label} failed: generation hit the token cap and the summary is incomplete`;
	}
	return undefined;
}

function createSummarizationOptions(
	model: Model<any>,
	maxTokens: number,
	apiKey: string | undefined,
	headers: Record<string, string> | undefined,
	env: Record<string, string> | undefined,
	signal: AbortSignal | undefined,
	thinkingLevel: ThinkingLevel | undefined,
	sessionId: string | undefined,
): SimpleStreamOptions {
	const options: SimpleStreamOptions = { maxTokens, signal, apiKey, headers, env, sessionId };
	if (model.reasoning && thinkingLevel && thinkingLevel !== "off") {
		options.reasoning = thinkingLevel;
	}
	return options;
}

/**
 * Shared choke point for every compaction/branch-summary summarization call. Wraps the
 * single LLM call in {@link retryAssistantCall} so transient stream drops (e.g.
 * `terminated`, socket close) honor the configured retry policy instead of failing
 * the whole compaction on the first attempt. Deterministic errors and aborts return
 * immediately (see {@link retryAssistantCall}).
 */
export async function completeSummarization(
	model: Model<any>,
	context: Context,
	options: SimpleStreamOptions,
	streamFn?: StreamFn,
	retry?: RetryPolicy,
	callbacks?: RetryCallbacks,
): Promise<AssistantMessage> {
	// Avoid cache writes for one-off summaries. Reuse caller-supplied routing when available;
	// callers without a session ID, including branch summaries, receive a fresh routing ID.
	const requestOptions: SimpleStreamOptions = {
		...options,
		cacheRetention: "none",
		sessionId: options.sessionId ?? uuidv7(),
	};
	const produce = async (): Promise<AssistantMessage> =>
		streamFn
			? (await streamFn(model, context, requestOptions)).result()
			: completeSimple(model, context, requestOptions);
	return retryAssistantCall(produce, retry, requestOptions.signal, callbacks);
}

/**
 * Generate a summary of the conversation using the LLM.
 * If previousSummary is provided, uses the update prompt to merge.
 */
export async function generateSummary(
	currentMessages: AgentMessage[],
	model: Model<any>,
	reserveTokens: number,
	apiKey: string | undefined,
	headers?: Record<string, string>,
	signal?: AbortSignal,
	customInstructions?: string,
	previousSummary?: string,
	thinkingLevel?: ThinkingLevel,
	streamFn?: StreamFn,
	env?: Record<string, string>,
	retry?: RetryPolicy,
	callbacks?: RetryCallbacks,
	sessionId?: string,
): Promise<string> {
	return (
		await generateSummaryWithUsage(
			currentMessages,
			model,
			reserveTokens,
			apiKey,
			headers,
			signal,
			customInstructions,
			previousSummary,
			thinkingLevel,
			streamFn,
			env,
			retry,
			callbacks,
			sessionId,
		)
	).text;
}

/** Build the provider context for a standalone summary request. */
function buildSummarizationContext(promptText: string): Context {
	return {
		systemPrompt: SUMMARIZATION_SYSTEM_PROMPT,
		messages: [
			{
				role: "user",
				content: [{ type: "text", text: promptText }],
				timestamp: Date.now(),
			},
		],
	};
}

/** Generate or update a conversation summary and return its provider usage. */
export async function generateSummaryWithUsage(
	currentMessages: AgentMessage[],
	model: Model<any>,
	reserveTokens: number,
	apiKey: string | undefined,
	headers?: Record<string, string>,
	signal?: AbortSignal,
	customInstructions?: string,
	previousSummary?: string,
	thinkingLevel?: ThinkingLevel,
	streamFn?: StreamFn,
	env?: Record<string, string>,
	retry?: RetryPolicy,
	callbacks?: RetryCallbacks,
	sessionId?: string,
): Promise<{ text: string; usage: Usage }> {
	const maxTokens = Math.min(
		Math.floor(0.8 * reserveTokens),
		model.maxTokens > 0 ? model.maxTokens : Number.POSITIVE_INFINITY,
	);

	// Use update prompt if we have a previous summary, otherwise initial prompt
	let basePrompt = previousSummary ? UPDATE_SUMMARIZATION_PROMPT : SUMMARIZATION_PROMPT;
	if (customInstructions) {
		basePrompt = `${basePrompt}\n\nAdditional focus: ${customInstructions}`;
	}

	// Serialize conversation to text so model doesn't try to continue it
	// Convert to LLM messages first (handles custom types like bashExecution, custom, etc.)
	const llmMessages = convertToLlm(currentMessages);
	const conversationText = serializeConversation(llmMessages);

	// Build the prompt with conversation wrapped in tags
	let promptText = `<conversation>\n${conversationText}\n</conversation>\n\n`;
	if (previousSummary) {
		promptText += `<previous-summary>\n${previousSummary}\n</previous-summary>\n\n`;
	}
	promptText += basePrompt;

	const completionOptions = createSummarizationOptions(
		model,
		maxTokens,
		apiKey,
		headers,
		env,
		signal,
		thinkingLevel,
		sessionId,
	);

	const response = await completeSummarization(
		model,
		buildSummarizationContext(promptText),
		completionOptions,
		streamFn,
		retry,
		callbacks,
	);

	const failure = getSummarizationFailure(response, "Summarization");
	if (failure) {
		throw new Error(failure);
	}
	if (response.content.some((block) => block.type === "toolCall")) {
		throw new Error("Summarization attempted to call a tool");
	}

	const textContent = contentText(response.content);

	return { text: textContent, usage: response.usage };
}

// ============================================================================
// Compaction Preparation (for extensions)
// ============================================================================

export interface CompactionPreparation {
	/** UUID of first entry to keep */
	firstKeptEntryId: string;
	/** Messages that will be summarized and discarded */
	messagesToSummarize: AgentMessage[];
	/** Messages that will be turned into turn prefix summary (if splitting) */
	turnPrefixMessages: AgentMessage[];
	/** Whether this is a split turn (cut point in middle of turn) */
	isSplitTurn: boolean;
	tokensBefore: number;
	/** One-shot anomaly message when the reported tokensBefore figure was rejected as implausible and this is the first such rejection for this context window - print through the caller's own UI-aware channel. */
	tokensBeforeWarning?: string;
	/** Summary from previous compaction, for iterative update */
	previousSummary?: string;
	/** File operations extracted from messagesToSummarize */
	fileOps: FileOperations;
	/** Compaction settions from settings.jsonl	*/
	settings: CompactionSettings;
}

export function prepareCompaction(
	pathEntries: SessionEntry[],
	settings: CompactionSettings,
	contextWindow?: number,
): CompactionPreparation | undefined {
	if (pathEntries.length > 0 && pathEntries[pathEntries.length - 1].type === "compaction") {
		return undefined;
	}

	let prevCompactionIndex = -1;
	for (let i = pathEntries.length - 1; i >= 0; i--) {
		if (pathEntries[i].type === "compaction") {
			prevCompactionIndex = i;
			break;
		}
	}

	let previousSummary: string | undefined;
	let boundaryStart = 0;
	if (prevCompactionIndex >= 0) {
		const prevCompaction = pathEntries[prevCompactionIndex] as CompactionEntry;
		previousSummary = prevCompaction.summary;
		const firstKeptEntryIndex = pathEntries.findIndex((entry) => entry.id === prevCompaction.firstKeptEntryId);
		boundaryStart = firstKeptEntryIndex >= 0 ? firstKeptEntryIndex : prevCompactionIndex + 1;
	}
	const boundaryEnd = pathEntries.length;

	const tokensBeforeEstimate = estimateContextTokens(buildSessionContext(pathEntries).messages, contextWindow);
	const tokensBefore = tokensBeforeEstimate.tokens;
	const tokensBeforeWarning = tokensBeforeEstimate.warning;

	const cutPoint = findCutPoint(pathEntries, boundaryStart, boundaryEnd, settings.keepRecentTokens);

	// Get UUID of first kept entry
	const firstKeptEntry = pathEntries[cutPoint.firstKeptEntryIndex];
	if (!firstKeptEntry?.id) {
		return undefined; // Session needs migration
	}
	const firstKeptEntryId = firstKeptEntry.id;

	const historyEnd = cutPoint.isSplitTurn ? cutPoint.turnStartIndex : cutPoint.firstKeptEntryIndex;

	// Messages to summarize (will be discarded after summary)
	const messagesToSummarize: AgentMessage[] = [];
	for (let i = boundaryStart; i < historyEnd; i++) {
		const msg = getMessageFromEntryForCompaction(pathEntries[i]);
		if (msg) messagesToSummarize.push(msg);
	}

	// Messages for turn prefix summary (if splitting a turn)
	const turnPrefixMessages: AgentMessage[] = [];
	if (cutPoint.isSplitTurn) {
		for (let i = cutPoint.turnStartIndex; i < cutPoint.firstKeptEntryIndex; i++) {
			const msg = getMessageFromEntryForCompaction(pathEntries[i]);
			if (msg) turnPrefixMessages.push(msg);
		}
	}

	if (messagesToSummarize.length === 0 && turnPrefixMessages.length === 0) {
		return undefined;
	}

	// Extract file operations from messages and previous compaction
	const fileOps = extractFileOperations(messagesToSummarize, pathEntries, prevCompactionIndex);

	// Also extract file ops from turn prefix if splitting
	if (cutPoint.isSplitTurn) {
		for (const msg of turnPrefixMessages) {
			extractFileOpsFromMessage(msg, fileOps);
		}
	}

	return {
		firstKeptEntryId,
		messagesToSummarize,
		turnPrefixMessages,
		isSplitTurn: cutPoint.isSplitTurn,
		tokensBefore,
		tokensBeforeWarning,
		previousSummary,
		fileOps,
		settings,
	};
}

// ============================================================================
// Main compaction function
// ============================================================================

const TURN_PREFIX_SUMMARIZATION_PROMPT = `This is the PREFIX of a turn that was too large to keep. The SUFFIX (recent work) is retained.

Summarize the prefix to provide context for the retained suffix:

## Original Request
[What did the user ask for in this turn?]

## Early Progress
- [Key decisions and work done in the prefix]

## Context for Suffix
- [Information needed to understand the retained recent work]

Be concise. Focus on what's needed to understand the kept suffix.`;

/**
 * Generate summaries for compaction using prepared data.
 * Returns CompactionResult - SessionManager adds uuid/parentUuid when saving.
 *
 * @param preparation - Pre-calculated preparation from prepareCompaction()
 * @param customInstructions - Optional custom focus for the summary
 * @param sessionId - Optional routing session ID forwarded without enabling prompt caching
 */
export async function compact(
	preparation: CompactionPreparation,
	model: Model<any>,
	apiKey: string | undefined,
	headers?: Record<string, string>,
	customInstructions?: string,
	signal?: AbortSignal,
	thinkingLevel?: ThinkingLevel,
	streamFn?: StreamFn,
	env?: Record<string, string>,
	retry?: RetryPolicy,
	callbacks?: RetryCallbacks,
	sessionId?: string,
): Promise<CompactionResult> {
	const {
		firstKeptEntryId,
		messagesToSummarize,
		turnPrefixMessages,
		isSplitTurn,
		tokensBefore,
		previousSummary,
		fileOps,
		settings,
	} = preparation;

	// Generate summaries and merge into one
	let summary: string;
	let summaryUsage: Usage;

	if (isSplitTurn && turnPrefixMessages.length > 0) {
		let historyText = "No prior history.";
		let historyUsage: Usage | undefined;
		if (messagesToSummarize.length > 0) {
			const historyResult = await generateSummaryWithUsage(
				messagesToSummarize,
				model,
				settings.reserveTokens,
				apiKey,
				headers,
				signal,
				customInstructions,
				previousSummary,
				thinkingLevel,
				streamFn,
				env,
				retry,
				callbacks,
				sessionId,
			);
			historyText = historyResult.text;
			historyUsage = historyResult.usage;
		}
		const turnPrefixResult = await generateTurnPrefixSummary(
			turnPrefixMessages,
			model,
			settings.reserveTokens,
			apiKey,
			headers,
			env,
			signal,
			thinkingLevel,
			streamFn,
			retry,
			callbacks,
			sessionId,
		);
		// Merge into single summary
		summary = `${historyText}\n\n---\n\n**Turn Context (split turn):**\n\n${turnPrefixResult.text}`;
		summaryUsage = historyUsage ? combineUsage(historyUsage, turnPrefixResult.usage) : turnPrefixResult.usage;
	} else {
		// Just generate history summary
		const result = await generateSummaryWithUsage(
			messagesToSummarize,
			model,
			settings.reserveTokens,
			apiKey,
			headers,
			signal,
			customInstructions,
			previousSummary,
			thinkingLevel,
			streamFn,
			env,
			retry,
			callbacks,
			sessionId,
		);
		summary = result.text;
		summaryUsage = result.usage;
	}

	// Compute file lists and append to summary
	const { readFiles, modifiedFiles } = computeFileLists(fileOps);
	summary += formatFileOperations(readFiles, modifiedFiles);

	if (!firstKeptEntryId) {
		throw new Error("First kept entry has no UUID - session may need migration");
	}

	return {
		summary,
		firstKeptEntryId,
		tokensBefore,
		usage: summaryUsage,
		details: { readFiles, modifiedFiles } as CompactionDetails,
	};
}

/**
 * Generate a summary for a turn prefix (when splitting a turn).
 */
async function generateTurnPrefixSummary(
	messages: AgentMessage[],
	model: Model<any>,
	reserveTokens: number,
	apiKey: string | undefined,
	headers?: Record<string, string>,
	env?: Record<string, string>,
	signal?: AbortSignal,
	thinkingLevel?: ThinkingLevel,
	streamFn?: StreamFn,
	retry?: RetryPolicy,
	callbacks?: RetryCallbacks,
	sessionId?: string,
): Promise<{ text: string; usage: Usage }> {
	const maxTokens = Math.min(
		Math.floor(0.5 * reserveTokens),
		model.maxTokens > 0 ? model.maxTokens : Number.POSITIVE_INFINITY,
	); // Smaller budget for turn prefix
	const llmMessages = convertToLlm(messages);
	const conversationText = serializeConversation(llmMessages);
	const promptText = `<conversation>\n${conversationText}\n</conversation>\n\n${TURN_PREFIX_SUMMARIZATION_PROMPT}`;

	const response = await completeSummarization(
		model,
		buildSummarizationContext(promptText),
		createSummarizationOptions(model, maxTokens, apiKey, headers, env, signal, thinkingLevel, sessionId),
		streamFn,
		retry,
		callbacks,
	);

	const failure = getSummarizationFailure(response, "Turn prefix summarization");
	if (failure) {
		throw new Error(failure);
	}
	if (response.content.some((block) => block.type === "toolCall")) {
		throw new Error("Turn prefix summarization attempted to call a tool");
	}

	return {
		text: contentText(response.content),
		usage: response.usage,
	};
}
