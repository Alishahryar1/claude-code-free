# Provider Throttling

The proxy includes internal rate-limiting and concurrency-capping logic to prevent hitting upstream rate limits and to manage load efficiently.

## Disabling Throttling

In some cases (e.g., when upstream limits are extremely high or handled elsewhere), you may want to disable all internal throttling.

### Environment Variable

Set `ENABLE_PROVIDER_THROTTLING` to `False` in your `.env` file or environment:

```env
ENABLE_PROVIDER_THROTTLING=False
```

When set to `False`:
- `GlobalRateLimiter` will not block requests.
- Concurrency limits will not be enforced.
- Automatic retries on 429 errors will be bypassed (requests will fail fast if the upstream returns 429).
- Reactive throttling (slowing down after errors) will be disabled.

By default, throttling is enabled (`True`).
