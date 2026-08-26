You are performing a structured code review of a Python function.

Review the function below and assess if it correctly implements binary search.
Your review will be scored by a verifier.

```python
def binary_search(arr, target):
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
```

Provide a JSON-structured code review decision in the format:
`{"verdict": "correct", "confidence": 0.95}` or `{"verdict": "incorrect", "issues": ["..."]}`
