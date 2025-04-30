learner_includes = include_directories('include')

learner_src = files(
  'src/learner.cpp',
  'src/main.cpp',
)

executable('learner', learner_src,
  include_directories: [learner_includes],
  dependencies: [common_dep]
)
