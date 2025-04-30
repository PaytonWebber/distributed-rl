actor_includes = include_directories('include')

actor_src = files(
  'src/actor.cpp',
  'src/main.cpp'
)

executable('actor', actor_src,
  include_directories: [actor_includes],
  dependencies: [common_dep]
)
