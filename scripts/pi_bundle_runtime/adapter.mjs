/** Exact Skill-set projection using only Pi's public SDK. The host must serialize
 * update() with prompt()/steer()/followUp(); updates run between turns. */
export function createBundleResources(pi, options) {
  const { additionalSkillPaths = [], ...rest } = options;
  let selected = [...additionalSkillPaths];
  const readBundle = (paths) => {
    const result = pi.loadSkills({
      cwd: rest.cwd, agentDir: rest.agentDir,
      skillPaths: paths, includeDefaults: false,
    });
    if (result.diagnostics.some((d) => d.type === 'error')) {
      throw new Error(JSON.stringify(result.diagnostics));
    }
    return result;
  };
  const loader = new pi.DefaultResourceLoader({
    ...rest, noSkills: true, additionalSkillPaths: [...selected],
    skillsOverride: () => readBundle(selected),
  });
  return {
    loader,
    async update(session, paths) {
      if (session.resourceLoader !== loader) throw new Error('Wrong session resource loader');
      if (session.isStreaming || session.isCompacting) throw new Error('Bundle update requires an idle session');
      const candidate = [...paths];
      // Validate before tearing down extensions. No mutation of Skill files.
      const result = readBundle(candidate);
      if (result.skills.length !== candidate.length) throw new Error('Expected one unique Skill per explicit path');
      selected = candidate;
      // A failure propagates: the host must not start another turn until recovered.
      await session.reload();
      return loader.getSkills();
    },
  };
}
