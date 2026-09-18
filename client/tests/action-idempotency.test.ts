import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { ApiClient } from '@/api/client';
import { ApiError } from '@/api/errors';
import { createFeatureApi } from '@/api/features';

describe('durable action request identity', () => {
  beforeEach(() => {
    sessionStorage.clear();
  });

  it('reuses the same key after a timeout and clears it after a confirmed response', async () => {
    const request = vi
      .fn()
      .mockRejectedValueOnce(new ApiError('timeout', 'timed out'))
      .mockResolvedValueOnce({ status: 'cancelled' })
      .mockResolvedValueOnce({ status: 'cancelled' });
    const api = createFeatureApi({ request } as unknown as ApiClient);

    await expect(api.cancelFeature('feature-1', 'done')).rejects.toMatchObject({ kind: 'timeout' });
    await api.cancelFeature('feature-1', 'done');
    await api.cancelFeature('feature-1', 'done');

    const first = request.mock.calls[0]?.[2].idempotencyKey;
    const recovered = request.mock.calls[1]?.[2].idempotencyKey;
    const laterDecision = request.mock.calls[2]?.[2].idempotencyKey;
    expect(recovered).toBe(first);
    expect(laterDecision).not.toBe(first);
  });
});
