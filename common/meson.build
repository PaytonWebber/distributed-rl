common_includes = include_directories('include')

common_src = files(
  'src/az_net.cpp',
  'src/residual_block.cpp'
)

torch_dep = dependency('Torch', required : true, method: 'cmake')

torch_libdir = run_command('python3', '-c', 'import torch; print(torch.__path__[0] + "/lib")').stdout().strip()
libtorch_cpu = torch_libdir + '/libtorch_cpu.so'
libtorch_cuda = torch_libdir + '/libtorch_cuda.so'

libzmq = dependency('libzmq', required: true)
nlohmann_json = dependency('nlohmann_json', required: true)
cppzmq_include = include_directories('/usr/include')

libtorch_dep = declare_dependency(
  link_args: [libtorch_cpu, libtorch_cuda]
)

common_lib = static_library('common', common_src,
  include_directories: [common_includes],
  dependencies: [torch_dep, libtorch_dep],
)

common_dep = declare_dependency(
  link_with: common_lib,
  include_directories: [common_includes, cppzmq_include],
  dependencies: [torch_dep, libtorch_dep, libzmq, nlohmann_json],
)
