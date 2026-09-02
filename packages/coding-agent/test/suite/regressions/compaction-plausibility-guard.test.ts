import { type AssistantMessage, fauxAssistantMessage, type Usage } from "@earendil-works/pi-ai";
import { afterEach, describe, expect, it, vi } from "vitest";
import { clearContextTokensWarningsForTests } from "../../../src/core/compaction/index.ts";
import type { CompactionEntry } from "../../../src/core/session-manager.ts";
import { createHarness, type Harness } from "../harness.ts";

// Reported shape: a session's `[compaction] Compacted from 1,782,723 tokens` entry on a
// model with a 1,000,000-token context window (~78% over the window, ~24x the session's
// actual stored content, in the same order of magnitude as an accumulating usage counter
// rather than a per-call figure), immediately followed by a second compaction attempt
// failing with "Nothing to compact (session too small)" because almost nothing was left
// after the bogus-triggered compaction. Traced to calculateContextTokens()/
// estimateContextTokens() trusting the provider-reported usage.totalTokens at face value
// with no sanity check against the model's own context window.

type SessionWithCompactionInternals = {
	_checkCompaction: (assistantMessage: AssistantMessage, skipAbortedCheck?: boolean) => Promise<boolean>;
	_runAutoCompaction: (reason: "overflow" | "threshold", willRetry: boolean) => Promise<boolean>;
};

// Scaled proportionally to the reported incident (1,782,723 tokens on a 1,000,000-token
// window, ~1.78x over) - matches the scaling convention used by the case3
// post-compaction-error-fallback regression test.
const CONTEXT_WINDOW = 1000;
const IMPLAUSIBLE_REPORTED_TOKENS = 1780;

function usageOf(totalTokens: number, input = totalTokens): Usage {
	return {
		input,
		output: 0,
		cacheRead: 0,
		cacheWrite: 0,
		totalTokens,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	};
}

/**
 * `input` defaults to a small, plausible figure independent of `totalTokens` - this mirrors
 * the reported incident's own evidence (component usage fields, e.g. input, looked ordinary;
 * only the reported totalTokens was the outlier, consistent with an accumulating counter) and
 * keeps these tests isolated to calculateContextTokens' plausibility guard rather than also
 * tripping `isContextOverflow`'s separate silent-overflow heuristic (packages/ai), which reads
 * usage.input + usage.cacheRead and is out of this fix's scope.
 */
function assistantWithUsage(
	harness: Harness,
	totalTokens: number,
	stopReason: AssistantMessage["stopReason"] = "stop",
	input = 50,
): AssistantMessage {
	const model = harness.getModel();
	return {
		role: "assistant",
		content: stopReason === "error" ? [] : [{ type: "text", text: "response" }],
		api: model.api,
		provider: model.provider,
		model: model.id,
		usage: usageOf(totalTokens, input),
		stopReason,
		timestamp: Date.now(),
	};
}

async function createGuardHarness(): Promise<Harness> {
	return createHarness({
		models: [{ id: "faux-1", contextWindow: CONTEXT_WINDOW, maxTokens: 200 }],
		settings: { compaction: { enabled: true, reserveTokens: 10 } },
	});
}

describe("compaction rejects an implausible reported context size (tokensBefore 1,782,723 on a 1M window)", () => {
	const harnesses: Harness[] = [];

	afterEach(() => {
		vi.restoreAllMocks();
		clearContextTokensWarningsForTests();
		while (harnesses.length > 0) {
			harnesses.pop()?.cleanup();
		}
	});

	describe("_checkCompaction trigger decision", () => {
		it("does not auto-compact from an implausible reported total when the real content is small", async () => {
			const harness = await createGuardHarness();
			harnesses.push(harness);
			const bogus = assistantWithUsage(harness, IMPLAUSIBLE_REPORTED_TOKENS);
			harness.session.agent.state.messages = [
				{ role: "user", content: [{ type: "text", text: "hi" }], timestamp: Date.now() - 1 },
				bogus,
			];
			const sessionInternals = harness.session as unknown as SessionWithCompactionInternals;
			const runAutoCompactionSpy = vi.spyOn(sessionInternals, "_runAutoCompaction").mockResolvedValue(false);

			await sessionInternals._checkCompaction(bogus);

			expect(runAutoCompactionSpy).not.toHaveBeenCalled();
		});

		it("still auto-compacts from a plausible reported total (no regression)", async () => {
			const harness = await createGuardHarness();
			harnesses.push(harness);
			// 995 > contextWindow(1000) - reserveTokens(10) = 990
			const plausible = assistantWithUsage(harness, 995);
			harness.session.agent.state.messages = [
				{ role: "user", content: [{ type: "text", text: "hi" }], timestamp: Date.now() - 1 },
				plausible,
			];
			const sessionInternals = harness.session as unknown as SessionWithCompactionInternals;
			const runAutoCompactionSpy = vi.spyOn(sessionInternals, "_runAutoCompaction").mockResolvedValue(false);

			await sessionInternals._checkCompaction(plausible);

			expect(runAutoCompactionSpy).toHaveBeenCalledWith("threshold", false);
		});

		it("surfaces the rejection once instead of compacting silently on it, even across repeated turns", async () => {
			const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
			const harness = await createGuardHarness();
			harnesses.push(harness);
			const bogus = assistantWithUsage(harness, IMPLAUSIBLE_REPORTED_TOKENS);
			harness.session.agent.state.messages = [
				{ role: "user", content: [{ type: "text", text: "hi" }], timestamp: Date.now() - 1 },
				bogus,
			];
			const sessionInternals = harness.session as unknown as SessionWithCompactionInternals;
			vi.spyOn(sessionInternals, "_runAutoCompaction").mockResolvedValue(false);

			await sessionInternals._checkCompaction(bogus);
			await sessionInternals._checkCompaction(bogus);
			await sessionInternals._checkCompaction(bogus);

			const anomalyWarnings = warnSpy.mock.calls.filter((call) => String(call[0]).includes("implausible"));
			expect(anomalyWarnings).toHaveLength(1);
			expect(String(anomalyWarnings[0][0])).toContain("1,000");
		});

		// The pre-prompt check (skipAbortedCheck: false) is the call site that lets an aborted
		// assistant message reach Case 3's else-branch with its own usage still attached. That
		// message is exactly one estimateContextTokens' last-valid-usage lookup skips, so a
		// guard keyed on that lookup would vet an older message - or none at all - and let the
		// anomalous reading through unclamped.
		it("does not auto-compact from an aborted turn's own implausible reported total", async () => {
			const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
			const harness = await createGuardHarness();
			harnesses.push(harness);
			const abortedBogus = assistantWithUsage(harness, IMPLAUSIBLE_REPORTED_TOKENS, "aborted");
			harness.session.agent.state.messages = [
				{ role: "user", content: [{ type: "text", text: "hi" }], timestamp: Date.now() - 1 },
				abortedBogus,
			];
			const sessionInternals = harness.session as unknown as SessionWithCompactionInternals;
			const runAutoCompactionSpy = vi.spyOn(sessionInternals, "_runAutoCompaction").mockResolvedValue(false);

			await sessionInternals._checkCompaction(abortedBogus, false);

			expect(runAutoCompactionSpy).not.toHaveBeenCalled();
			expect(warnSpy.mock.calls.filter((call) => String(call[0]).includes("implausible"))).toHaveLength(1);
		});

		it("rejects an aborted turn's implausible total even when an earlier plausible usage exists to be mistaken for it", async () => {
			const harness = await createGuardHarness();
			harnesses.push(harness);
			// A plausible, well-under-threshold baseline: the message estimateContextTokens'
			// own lookup would settle on once it skips the aborted turn.
			const baseline = assistantWithUsage(harness, 68, "stop");
			const abortedBogus = assistantWithUsage(harness, IMPLAUSIBLE_REPORTED_TOKENS, "aborted");
			harness.session.agent.state.messages = [
				{ role: "user", content: [{ type: "text", text: "hi" }], timestamp: Date.now() - 3 },
				baseline,
				{ role: "user", content: [{ type: "text", text: "next" }], timestamp: Date.now() - 1 },
				abortedBogus,
			];
			const sessionInternals = harness.session as unknown as SessionWithCompactionInternals;
			const runAutoCompactionSpy = vi.spyOn(sessionInternals, "_runAutoCompaction").mockResolvedValue(false);

			await sessionInternals._checkCompaction(abortedBogus, false);

			expect(runAutoCompactionSpy).not.toHaveBeenCalled();
		});

		it("still auto-compacts from an aborted turn's plausible reported total (no regression)", async () => {
			const harness = await createGuardHarness();
			harnesses.push(harness);
			// 995 > contextWindow(1000) - reserveTokens(10) = 990
			const abortedPlausible = assistantWithUsage(harness, 995, "aborted");
			harness.session.agent.state.messages = [
				{ role: "user", content: [{ type: "text", text: "hi" }], timestamp: Date.now() - 1 },
				abortedPlausible,
			];
			const sessionInternals = harness.session as unknown as SessionWithCompactionInternals;
			const runAutoCompactionSpy = vi.spyOn(sessionInternals, "_runAutoCompaction").mockResolvedValue(false);

			await sessionInternals._checkCompaction(abortedPlausible, false);

			expect(runAutoCompactionSpy).toHaveBeenCalledWith("threshold", false);
		});

		// Case 3's if-branch: a later turn failing with stopReason "error" (e.g. a 529) falls
		// back to the last valid usage - which is the anomalous reading from the turn before.
		// Clamping that to the window rather than rejecting it pins contextTokens to exactly
		// the window, which is always over the threshold, so the bogus compaction fires anyway.
		it("does not auto-compact when a failed turn falls back to an earlier turn's implausible reported total", async () => {
			const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
			const harness = await createGuardHarness();
			harnesses.push(harness);
			const bogus = assistantWithUsage(harness, IMPLAUSIBLE_REPORTED_TOKENS, "stop");
			const failedTurn = assistantWithUsage(harness, 0, "error", 0);
			harness.session.agent.state.messages = [
				{ role: "user", content: [{ type: "text", text: "hi" }], timestamp: Date.now() - 3 },
				bogus,
				{ role: "user", content: [{ type: "text", text: "next" }], timestamp: Date.now() - 1 },
				failedTurn,
			];
			const sessionInternals = harness.session as unknown as SessionWithCompactionInternals;
			const runAutoCompactionSpy = vi.spyOn(sessionInternals, "_runAutoCompaction").mockResolvedValue(false);

			await sessionInternals._checkCompaction(failedTurn);

			expect(runAutoCompactionSpy).not.toHaveBeenCalled();
			const anomalyWarnings = warnSpy.mock.calls.filter((call) => String(call[0]).includes("implausible"));
			expect(anomalyWarnings).toHaveLength(1);
			expect(String(anomalyWarnings[0][0])).toContain("1,000");
		});
	});

	describe("manual compact() end to end", () => {
		function seedSession(harness: Harness, totalTokens: number): void {
			harness.settingsManager.applyOverrides({ compaction: { keepRecentTokens: 1 } });
			const now = Date.now();
			harness.sessionManager.appendMessage({
				role: "user",
				content: [{ type: "text", text: "message to compact" }],
				timestamp: now - 1000,
			});
			const model = harness.getModel();
			const assistant: AssistantMessage = {
				role: "assistant",
				content: [{ type: "text", text: "assistant response to compact" }],
				api: model.api,
				provider: model.provider,
				model: model.id,
				usage: usageOf(totalTokens),
				stopReason: "stop",
				timestamp: now - 500,
			};
			harness.sessionManager.appendMessage(assistant);
			harness.session.agent.state.messages = harness.sessionManager.buildSessionContext().messages;
		}

		it("compacts correctly end to end when the reported value is plausible (no regression)", async () => {
			const harness = await createGuardHarness();
			harnesses.push(harness);
			seedSession(harness, 500); // well within the 1000-token window
			harness.setResponses([fauxAssistantMessage("summary of prior work")]);

			const result = await harness.session.compact();

			expect(result.tokensBefore).toBeGreaterThan(0);
			expect(result.tokensBefore).toBeLessThanOrEqual(CONTEXT_WINDOW);
			const compactionEntries = harness.sessionManager.getEntries().filter((entry) => entry.type === "compaction");
			expect(compactionEntries).toHaveLength(1);
			expect((compactionEntries[0] as CompactionEntry).tokensBefore).toBe(result.tokensBefore);
		});

		it("rejects an implausible reported tokensBefore and still compacts end to end from the real content", async () => {
			const harness = await createGuardHarness();
			harnesses.push(harness);
			seedSession(harness, IMPLAUSIBLE_REPORTED_TOKENS); // mirrors the reported 1,782,723 on a 1,000,000 window
			harness.setResponses([fauxAssistantMessage("summary of prior work")]);

			const result = await harness.session.compact();

			// The persisted tokensBefore must never be the physically-impossible reported figure.
			expect(result.tokensBefore).not.toBe(IMPLAUSIBLE_REPORTED_TOKENS);
			expect(result.tokensBefore).toBeGreaterThan(0);
			expect(result.tokensBefore).toBeLessThan(CONTEXT_WINDOW);
			const compactionEntries = harness.sessionManager.getEntries().filter((entry) => entry.type === "compaction");
			expect(compactionEntries).toHaveLength(1);
			expect((compactionEntries[0] as CompactionEntry).tokensBefore).toBe(result.tokensBefore);
		});
	});
});
