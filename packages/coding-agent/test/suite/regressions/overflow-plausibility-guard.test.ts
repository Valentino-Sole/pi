import type { AssistantMessage, Usage } from "@earendil-works/pi-ai";
import { afterEach, describe, expect, it, vi } from "vitest";
import { clearContextTokensWarningsForTests } from "../../../src/core/compaction/index.ts";
import { createHarness, type Harness } from "../harness.ts";

// isContextOverflow() (packages/ai/src/utils/overflow.ts) runs ahead of Case 3's plausibility
// guard (see compaction-plausibility-guard.test.ts) and, for its z.ai-style "silent overflow"
// case, judges the exact same locally-reported usage.input + usage.cacheRead figure Case 3 was
// hardened against - a figure that has been observed in production to wildly overstate a
// session's real content (819,151 reported vs ~85,600 real measured content, ~9.6x). Left as a
// documented residual risk in the original fix (fm/pi-compaction-plausibility-guard
// round2-review-findings, item overflow-branch-precedes-guard) because packages/ai has no access
// to message history and touching its shared overflow contract would affect every other
// provider. This closes that gap at the one call site that has message history available
// (agent-session.ts's `_checkCompaction`), leaving overflow.ts itself untouched.
//
// isContextOverflow's third case (Xiaomi MiMo-style length stop) is intentionally not gated:
// isRecoverableLength() independently fires on exactly those messages (stopReason "length" with
// zero output is always below a positive maxTokens) and drives the same compact-and-retry
// branch, so rejecting the overflow classification there would change no observable behavior.

type SessionWithCompactionInternals = {
	_checkCompaction: (assistantMessage: AssistantMessage, skipAbortedCheck?: boolean) => Promise<boolean>;
	_runAutoCompaction: (reason: "overflow" | "threshold", willRetry: boolean) => Promise<boolean>;
};

const CONTEXT_WINDOW = 1_000_000;

function usageOf(input: number, cacheRead = 0, output = 0): Usage {
	return {
		input,
		output,
		cacheRead,
		cacheWrite: 0,
		totalTokens: input + output + cacheRead,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	};
}

function overflowAssistantMessage(
	harness: Harness,
	usage: Usage,
	stopReason: AssistantMessage["stopReason"],
): AssistantMessage {
	const model = harness.getModel();
	return {
		role: "assistant",
		content: stopReason === "error" ? [] : [{ type: "text", text: "response" }],
		api: model.api,
		provider: model.provider,
		model: model.id,
		usage,
		stopReason,
		errorMessage: stopReason === "error" ? "prompt is too long: 1782723 tokens > 1000000 maximum" : undefined,
		timestamp: Date.now(),
	};
}

async function createGuardHarness(): Promise<Harness> {
	return createHarness({
		models: [{ id: "faux-1", contextWindow: CONTEXT_WINDOW, maxTokens: 200 }],
		settings: { compaction: { enabled: true, reserveTokens: 10 } },
	});
}

// The baseline the guard judges against: the session's own last valid reported usage before the
// message under judgement. Deliberately not a message-size sum - see resolveOverflowPlausibility.
function priorTurn(harness: Harness, reportedTokens: number): AssistantMessage {
	return {
		...overflowAssistantMessage(harness, usageOf(reportedTokens), "stop"),
		timestamp: Date.now() - 2,
	};
}

describe("isContextOverflow's silent-overflow signal rejects an implausible reported usage figure", () => {
	const harnesses: Harness[] = [];

	afterEach(() => {
		vi.restoreAllMocks();
		clearContextTokensWarningsForTests();
		while (harnesses.length > 0) {
			harnesses.pop()?.cleanup();
		}
	});

	// Mirrors the reported incident (1,782,723 reported, ~78% over a 1,000,000-token window)
	// but on a successful response (stopReason "stop"), which is exactly isContextOverflow's
	// z.ai-style silent-overflow case: usage.input + usage.cacheRead > contextWindow with no
	// error at all.
	it("does not auto-compact via the overflow path from a silent-overflow reading that wildly overstates the real measured content", async () => {
		const harness = await createGuardHarness();
		harnesses.push(harness);
		const bogus = overflowAssistantMessage(harness, usageOf(1_782_723), "stop");
		harness.session.agent.state.messages = [
			{ role: "user", content: [{ type: "text", text: "hello" }], timestamp: Date.now() - 3 },
			priorTurn(harness, 85_600),
			bogus,
		];
		const sessionInternals = harness.session as unknown as SessionWithCompactionInternals;
		const runAutoCompactionSpy = vi.spyOn(sessionInternals, "_runAutoCompaction").mockResolvedValue(false);

		await sessionInternals._checkCompaction(bogus);

		expect(runAutoCompactionSpy).not.toHaveBeenCalledWith("overflow", expect.anything());
	});

	it("surfaces the rejection once through the shared one-shot warning", async () => {
		const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
		const harness = await createGuardHarness();
		harnesses.push(harness);
		const bogus = overflowAssistantMessage(harness, usageOf(1_782_723), "stop");
		harness.session.agent.state.messages = [
			{ role: "user", content: [{ type: "text", text: "hello" }], timestamp: Date.now() - 3 },
			priorTurn(harness, 85_600),
			bogus,
		];
		const sessionInternals = harness.session as unknown as SessionWithCompactionInternals;
		vi.spyOn(sessionInternals, "_runAutoCompaction").mockResolvedValue(false);

		await sessionInternals._checkCompaction(bogus);

		const anomalyWarnings = warnSpy.mock.calls.filter((call) => String(call[0]).includes("implausible"));
		expect(anomalyWarnings).toHaveLength(1);
		expect(String(anomalyWarnings[0][0])).toContain("1,782,723");
	});

	// A genuine silent overflow on a tool-heavy session: the conversation's own messages measure
	// only ~85,600 tokens because the system prompt and tool definitions - which no message-size
	// sum can see - carry the rest, and the session has been legitimately reporting ~980,500
	// tokens all along. This must still compact-without-retry exactly as before.
	it("still auto-compacts via the overflow path when the reading tracks the session's own previous reported usage (no regression)", async () => {
		const harness = await createGuardHarness();
		harnesses.push(harness);
		const genuine = overflowAssistantMessage(harness, usageOf(1_050_000), "stop");
		harness.session.agent.state.messages = [
			{ role: "user", content: [{ type: "text", text: "x".repeat(342_400) }], timestamp: Date.now() - 3 }, // measures to 85,600
			priorTurn(harness, 980_500),
			genuine,
		];
		const sessionInternals = harness.session as unknown as SessionWithCompactionInternals;
		const runAutoCompactionSpy = vi.spyOn(sessionInternals, "_runAutoCompaction").mockResolvedValue(false);

		await sessionInternals._checkCompaction(genuine);

		expect(runAutoCompactionSpy).toHaveBeenCalledWith("overflow", false);
	});

	// Case 1 (a provider explicitly erroring "prompt is too long") is the provider's own
	// authoritative refusal, not a locally-computed figure - it must be trusted unconditionally,
	// even when the usage numbers attached to the same message look implausible.
	it("still trusts an error-message overflow signal even when its usage numbers look implausible", async () => {
		const harness = await createGuardHarness();
		harnesses.push(harness);
		const errored = overflowAssistantMessage(harness, usageOf(1_782_723), "error");
		harness.session.agent.state.messages = [
			{ role: "user", content: [{ type: "text", text: "hello" }], timestamp: Date.now() - 3 },
			priorTurn(harness, 85_600),
			errored,
		];
		const sessionInternals = harness.session as unknown as SessionWithCompactionInternals;
		const runAutoCompactionSpy = vi.spyOn(sessionInternals, "_runAutoCompaction").mockResolvedValue(false);

		await sessionInternals._checkCompaction(errored);

		expect(runAutoCompactionSpy).toHaveBeenCalledWith("overflow", true);
	});
});
