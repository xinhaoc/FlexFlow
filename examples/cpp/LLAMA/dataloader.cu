#include "flexflow/utils/cuda_helper.h"
#include "llama.h"

void DataLoader::load_input(Task const *task,
                            std::vector<PhysicalRegion> const &regions,
                            Context ctx,
                            Runtime *runtime) {
  fprintf(stderr, "----------start load input--------------");                                        
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  SampleIdxs *meta = (SampleIdxs *)task->local_args;

  TensorAccessorR<long, 3> full_input(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorW<long, 3> batch_input(regions[1],
                                            task->regions[1],
                                            FID_DATA,
                                            ctx,
                                            runtime,
                                            false /*readOutput*/);

  Domain full_input_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Domain batch_input_domain = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());

  coord_t token_dim =
      batch_input_domain.hi()[0] - batch_input_domain.lo()[0] + 1;
  coord_t sequence_length =
      batch_input_domain.hi()[1] - batch_input_domain.lo()[1] + 1;
  coord_t batch_size =
      batch_input_domain.hi()[2] - batch_input_domain.lo()[2] + 1;
  
   std::cout << "dims "<<"\n";
   std::cout << token_dim <<"\n";
   std::cout << sequence_length<<"\n";
   std::cout << batch_size<<"\n";

   std::cout << meta->num_samples<<"\n";
   
  

  // FIXME: currently assume continous indices
  assert(meta->num_samples <= batch_size);
  for (int i = 1; i < meta->num_samples; i++) {
    assert(meta->idxs[i] == meta->idxs[0] + i);
  }
  // pad inputs if needed (this is really only useful for debugging)
  if (meta->num_samples < batch_size) {
    checkCUDA(cudaMemset(batch_input.ptr +
                             token_dim * sequence_length * meta->num_samples,
                         0,
                         token_dim * sequence_length *
                             (batch_size - meta->num_samples) * sizeof(long)));
  }
  coord_t start_idx = meta->idxs[0];
  assert(batch_input_domain.get_volume() % token_dim * sequence_length *
             batch_size ==
         0);
  assert(batch_input_domain.get_volume() % batch_size == 0);
  size_t size_to_copy =
      (batch_input_domain.get_volume() / batch_size) * meta->num_samples;
  long const *input_zc =
      full_input.ptr + start_idx * token_dim * sequence_length;
  copy_kernel<<<GET_BLOCKS(size_to_copy), CUDA_NUM_THREADS>>>(
      batch_input.ptr, input_zc, size_to_copy);
  checkCUDA(cudaDeviceSynchronize());
}