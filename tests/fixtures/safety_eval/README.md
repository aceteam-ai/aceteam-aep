# Safety evaluation fixtures

This directory contains 400 original synthetic rows and five original policy rules. No row comes from a customer, platform trace, model output, or public dataset. All labels are author labels. The empty `votes` arrays mean every test row is unverified and must not be used for a headline calibration claim.

Each of the five categories has 40 violations, 20 hard negatives, and 20 easy negatives. `ethics` is held out entirely in `test_xd`. The other synthetic rows use eight authored template groups per category and kind; alternating groups go to `dev` and `test_id`, with equal counts for each kind. Each group has distinct fictional scene variants, and no normalized template crosses a split. Run `python scripts/safety_eval/generate_fixtures.py` to regenerate the catalog and rows, then `python -m aceteam_aep.safety.eval splits --labeled tests/fixtures/safety_eval/labeled.jsonl --check` to verify the split column.

The loader rejects duplicate ids, unknown policies, mismatched labels, normalized text shared across splits, nonreserved synthetic contact data, and FuzzyBench rows in test splits. It drops test rows whose two verifier votes disagree. `verification_summary` reports the disagreement drop rate and how many test rows remain unverified.

The `verify` command requires an explicitly supplied `module:attribute` client adapter and two model ids from different families than the generator. This slice tests it with a fake adapter only. No model verification has been run or paid for.

The design calls for a pinned `deepset/prompt-injections` subset as a source shift slice. No such rows are present yet because this slice is being built without network access or a locally vetted copy. The NOTICE records this provenance boundary. Add public rows only with their exact revision and license attribution.
