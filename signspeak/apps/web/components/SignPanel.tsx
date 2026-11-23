"use client";

/**
 * The signer's view of what the machine thinks they said.
 *
 * This panel is the heart of the confidence-aware UX in the spec, and it has
 * one job: never let the system's guess look like the signer's word. The
 * person whose voice is being synthesised has to be able to see, at a glance,
 * whether the machine got it right - and fix it when it did not.
 *
 * Three states, three visibly different treatments:
 *
 *   speak    high confidence, already spoken aloud - shown with a correction
 *            affordance, because "already spoken" is not "correct"
 *   confirm  ambiguous - shown as a question, nothing has been said yet
 *   reject   nothing usable - shown as a prompt to repeat, not as an error
 *
 * A "reject" is never framed as the user's mistake. "Didn't catch that" puts
 * the failure where it belongs: on the model.
 */

import { confidenceBand, type Decision } from "@/lib/decision";

interface Props {
  decision: Decision | null;
  signing: boolean;
  tracked: boolean;
  latency: { inference: number; pipeline: number } | null;
  vocab: string[];
  onCorrect: (actualLabel: string) => void;
}

export default function SignPanel({
  decision,
  signing,
  tracked,
  latency,
  vocab,
  onCorrect,
}: Props) {
  const band = decision ? confidenceBand(decision.confidence) : null;
  const toneClass =
    band?.tone === "high" ? "text-high" : band?.tone === "medium" ? "text-medium" : "text-low";

  return (
    <section
      aria-label="Sign recognition"
      className="rounded-xl border border-border bg-surface p-4"
    >
      <header className="mb-3 flex items-center justify-between">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-muted">
          Sign recognition
        </h2>
        <StatusDot signing={signing} tracked={tracked} />
      </header>

      {!tracked && (
        <p className="mb-3 rounded-md border border-medium/40 bg-medium/10 p-2 text-sm text-medium">
          Can&apos;t see your upper body. Move back so your head and both shoulders are in frame.
        </p>
      )}

      {!decision && (
        <p className="text-sm text-muted">
          Sign within the frame, then pause for about a third of a second. The pause is how the
          system knows the sign has finished.
        </p>
      )}

      {decision && (
        <>
          <div className="mb-3">
            <p className="mb-1 text-3xl font-semibold leading-tight">
              {decision.action === "reject" ? (
                <span className="text-muted">Didn&apos;t catch that</span>
              ) : (
                decision.label.replace(/_/g, " ")
              )}
            </p>

            {decision.action === "speak" && (
              <p className="text-sm text-muted">
                Spoken as “{decision.spoken}”
              </p>
            )}
            {decision.action === "confirm" && (
              <p className="text-sm text-medium">
                Not sure — sign again to confirm. Nothing has been said yet.
              </p>
            )}
            {decision.action === "reject" && (
              <p className="text-sm text-muted">{decision.reason}. Please sign again.</p>
            )}
          </div>

          {/* Confidence as number AND bar AND word. Colour alone fails for
              anyone with colour vision deficiency; a bar alone is imprecise;
              a number alone is easy to misread under time pressure. */}
          <div className="mb-3">
            <div className="mb-1 flex items-baseline justify-between text-sm">
              <span className={toneClass}>{band?.label}</span>
              <span className="tabular-nums text-muted">
                {(decision.confidence * 100).toFixed(0)}%
              </span>
            </div>
            <div className="confidence-track" role="img"
                 aria-label={`Confidence ${(decision.confidence * 100).toFixed(0)} percent, ${band?.label}`}>
              <div
                className="confidence-fill"
                style={{
                  width: `${Math.round(decision.confidence * 100)}%`,
                  background:
                    band?.tone === "high"
                      ? "rgb(var(--high))"
                      : band?.tone === "medium"
                        ? "rgb(var(--medium))"
                        : "rgb(var(--low))",
                }}
              />
            </div>
          </div>

          {decision.runnerUp && decision.runnerUpConfidence > 0.08 && (
            <p className="mb-3 text-xs text-muted">
              Next closest: {decision.runnerUp.replace(/_/g, " ")} (
              {(decision.runnerUpConfidence * 100).toFixed(0)}%)
            </p>
          )}

          {/* Corrections are the most valuable data this system produces: a
              labelled error from the real world, which no offline test set
              contains. Make them one tap. */}
          {decision.action !== "reject" && (
            <details className="rounded-md border border-border p-2">
              <summary className="cursor-pointer text-sm text-muted">
                Wrong? Tell it what you meant
              </summary>
              <div className="mt-2 flex flex-wrap gap-2">
                {vocab
                  .filter((v) => v !== "NONE" && v !== decision.label)
                  .map((v) => (
                    <button
                      key={v}
                      onClick={() => onCorrect(v)}
                      className="rounded-md border border-border px-2 py-1 text-xs hover:border-accent hover:text-accent"
                    >
                      {v.replace(/_/g, " ")}
                    </button>
                  ))}
              </div>
            </details>
          )}
        </>
      )}

      {latency && (
        <p className="mt-3 border-t border-border pt-2 text-xs tabular-nums text-muted">
          inference {latency.inference.toFixed(1)}ms · pipeline {latency.pipeline.toFixed(1)}ms
          <br />
          <span className="opacity-70">
            Excludes the ~280ms the segmenter waits to be sure the sign ended — the largest
            term in the end-to-end delay.
          </span>
        </p>
      )}
    </section>
  );
}

function StatusDot({ signing, tracked }: { signing: boolean; tracked: boolean }) {
  const label = !tracked ? "Not tracking" : signing ? "Reading sign" : "Watching";
  const color = !tracked ? "bg-low" : signing ? "bg-accent" : "bg-high";
  return (
    <span className="flex items-center gap-2 text-xs text-muted" role="status" aria-live="polite">
      <span className={`h-2.5 w-2.5 rounded-full ${color} ${signing ? "pulse-recording" : ""}`} />
      {label}
    </span>
  );
}
