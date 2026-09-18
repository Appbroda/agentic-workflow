import { describe, expect, it } from 'vitest';
import { setupStateSchema } from '@/schemas/feature';
import setupPerformanceTiers from './fixtures/setup.performance-tiers.json';

/**
 * What `//setup` really sends, parsed by the schema the client parses production with.
 *
 * The payload beside this test is a saved response from the platform's own handler with both
 * providers' tier presets configured — not a shape written from memory. A hand-written
 * fixture contains exactly the fields its author remembered, which is how this client
 * previously lost a dozen of them. Refresh it by re-running the capture against a configured
 * server.
 */

describe('the setup payload the platform sends', () => {
  it('parses six labeled (platform, tier) options with their resolved models', () => {
    const setup = setupStateSchema.parse(setupPerformanceTiers);

    expect(
      setup.agent_platforms.map((item) => [item.platform, item.performance_tier, item.label]),
    ).toEqual([
      ['openai', 'low', 'OpenAI — Economy'],
      ['openai', 'medium', 'OpenAI — Standard'],
      ['openai', 'high', 'OpenAI — Max'],
      ['anthropic', 'low', 'Claude — Economy'],
      ['anthropic', 'medium', 'Claude — Standard'],
      ['anthropic', 'high', 'Claude — Max'],
    ]);
    expect(setup.agent_platforms.every((item) => item.configured)).toBe(true);
    // The models come from the server so the form can say what a choice runs on without
    // holding a second copy of deployment configuration.
    const economy = setup.agent_platforms.find(
      (item) => item.platform === 'anthropic' && item.performance_tier === 'low',
    );
    expect(economy?.models).toEqual({
      reasoning: 'claude-sonnet-5',
      coding: 'claude-sonnet-5',
      review: 'claude-sonnet-5',
      scoped_fix: 'claude-haiku-4-5',
    });
    // Every role of every option resolves to something; an option that named no model would
    // be an option that fails at its first model call, minutes later, on a worker.
    for (const option of setup.agent_platforms) {
      expect(Object.keys(option.models).sort()).toEqual([
        'coding',
        'reasoning',
        'review',
        'scoped_fix',
      ]);
      // The efforts are keyed identically, so the form can read the two together.
      expect(Object.keys(option.reasoning_efforts).sort()).toEqual(Object.keys(option.models).sort());
    }
    // The effort each role is sent at, which the models alone do not say: Standard and Max
    // run reasoning on the same model and differ only here.
    const byKey = new Map(
      setup.agent_platforms.map((item) => [`${item.platform}:${item.performance_tier}`, item]),
    );
    expect(byKey.get('openai:medium')?.models.reasoning).toBe(
      byKey.get('openai:high')?.models.reasoning,
    );
    expect(byKey.get('openai:medium')?.reasoning_efforts.reasoning).toBe('high');
    expect(byKey.get('openai:high')?.reasoning_efforts.reasoning).toBe('max');
    // A role whose model takes no effort is published as `null` — the provider's own
    // default, stated, rather than a key the form has to guess the meaning of.
    expect(economy?.models.scoped_fix).toBe('claude-haiku-4-5');
    expect(economy?.reasoning_efforts.scoped_fix).toBeNull();
  });

  it('reads an entry from a server that predates tiers as the high tier', () => {
    // Everything such a server offers is the unsuffixed configuration, which is the high
    // tier — so the client names it rather than leaving the field undefined and guessing.
    const setup = setupStateSchema.parse({
      credentials_ready: true,
      providers: [],
      agent_platforms: [
        { platform: 'openai', label: 'OpenAI', configured: true, models: { coding: 'gpt-5.6-sol' } },
      ],
      repositories_ready: true,
      saved_repository_count: 1,
      credential_storage_available: true,
    });

    expect(setup.agent_platforms[0]!.performance_tier).toBe('high');
    // And it says nothing about effort, which is not the same as saying there is none: the
    // map is empty, so the form renders those stages as unreported rather than as defaults.
    expect(setup.agent_platforms[0]!.reasoning_efforts).toEqual({});
  });
});
