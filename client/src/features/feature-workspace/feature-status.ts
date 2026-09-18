import type { Feature } from '@/schemas/feature';

/**
 * The phase a person should act on.
 *
 * `status` is the durable recovery checkpoint. During asynchronous resume/retry work the API
 * keeps that checkpoint intact and composes its active queue intent into `effective_status`.
 */
export function effectiveFeatureStatus(
  feature: Pick<Feature, 'status' | 'effective_status'>,
): string {
  return feature.effective_status ?? feature.status;
}
