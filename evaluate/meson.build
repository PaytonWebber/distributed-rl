evaluate_includes = include_directories('include')

evaluate_src = files(
  'src/evaluate.cpp',
)

executable('evaluate', evaluate_src,
  include_directories: [evaluate_includes],
  dependencies: [common_dep]
)
