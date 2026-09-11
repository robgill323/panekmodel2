# story-3a verification: the silent-progress defect, before and after

The smoke run recorded in `smoke-run-2026-09-10.md` found that with entity
detection on — the default — every progress stage reported `done` about a
minute into a 324.6 s run, leaving **84% of the wall clock with no sign of
life**. NLTK NER ran as a trailing pass over the whole batch, under no stage.

## What changed

Entity detection moved out of the trailing pass and into the per-video
preparation, alongside fetch/chunk/embed/sentiment. Two consequences:

1. It can be reported. `entities` is now a stage in the progress contract.
2. It is cached. NER was by far the slowest step, and it used to re-run on
   every batch even when every video was a cache hit.

The stage list is also now in true execution order. It previously claimed
topic modelling ran before sentiment, which was never true — everything up to
entities runs per video, and the shared topic fit is genuinely last.

## Measured, on real videos through the server API

Cold run (cache miss forced by a chunk-size change, so nothing was deleted):

```
fetch      done at +3.0s
chunk      done at +24.3s
embed      done at +24.3s
sentiment  done at +24.3s
entities   done at +24.3s
topics     done at +28.3s
TOTAL 28.3s   silent tail: 0.0s (0%)
completion order monotone vs declared order: True
```

**Silent tail: 84% → 0%.**

Warm run, same two videos as the original smoke (625 chunks, 9 topics):

```
TOTAL WALL CLOCK: 12.4s   (was 324.6s cold)
```

A **26× speedup** on a repeat run, because entities now come from the cache
with everything else. That was not the goal of the task — it is a consequence
of fixing the defect at its cause rather than only reporting it better.

## A blemish found and fixed during verification

The first version marked `sentiment` done when the entities message arrived,
which made a single-video run show `sentiment done at +8.1s` and `chunk done
at +30.4s` — stages appearing to complete out of order. With a batch it was
also simply wrong, since the next video's chunking still lay ahead. Per-video
stages are now closed together when the shared fit begins. The check above
asserts the completion order matches the declared order.

## Stage counters

Rendering the finished progress screen exposed a second, smaller dishonesty:
every per-stage note still read `0/5 URLs`, `0/5 videos embedded` and so on,
because the counters were derived from per-URL outcome state that is only
written after the run returns. They never advanced during a run and were
wrong at the end of one. Counters are now driven by the per-video progress
messages themselves, and the per-video stages close with their true totals:

```
fetch      5/5 URLs
chunk      5/5 videos
embed      5/5 videos embedded
sentiment  5/5 videos scored
entities   5/5 videos
```

Pinned by two tests that drive the message sequence directly, rather than by
polling a fake runner that finishes between polls.

## Not covered

One machine, two videos. The 26× warm figure is specific to this corpus —
NER cost scales with chunk count, so the speedup is large here because one
video is 4.5 hours long. Whisper remains unexercised.
