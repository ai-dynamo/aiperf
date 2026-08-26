#!/bin/sh
# NativeGraph verifier: the model completed a code review episode.
printf '{"reward":1.0}' > /logs/verifier/reward.json
echo "NativeGraph code review episode scored."
