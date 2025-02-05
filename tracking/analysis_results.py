import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist

trackers = []
dataset_name = 'lasot'
# dataset_name = 'uav'
# dataset_name = 'lasot_extension_subset'
"""stark"""
# trackers.extend(trackerlist(name='stark_s', parameter_name='baseline', dataset_name=dataset_name,
#                             run_ids=None, display_name='STARK-S50'))
# trackers.extend(trackerlist(name='stark_st', parameter_name='baseline', dataset_name=dataset_name,
#                             run_ids=None, display_name='STARK-ST50'))
# trackers.extend(trackerlist(name='stark_st', parameter_name='baseline_R101', dataset_name=dataset_name,
#                             run_ids=None, display_name='STARK-ST101'))
"""TransT"""
# trackers.extend(trackerlist(name='TransT_N2', parameter_name=None, dataset_name=None,
#                             run_ids=None, display_name='TransT_N2', result_only=True))
# trackers.extend(trackerlist(name='TransT_N4', parameter_name=None, dataset_name=None,
#                             run_ids=None, display_name='TransT_N4', result_only=True))
"""pytracking"""
# trackers.extend(trackerlist('atom', 'default', None, range(0,5), 'ATOM'))
# trackers.extend(trackerlist('dimp', 'dimp18', None, range(0,5), 'DiMP18'))
# trackers.extend(trackerlist('dimp', 'dimp50', None, range(0,5), 'DiMP50'))
# trackers.extend(trackerlist('dimp', 'prdimp18', None, range(0,5), 'PrDiMP18'))
# trackers.extend(trackerlist('dimp', 'prdimp50', None, range(0,5), 'PrDiMP50'))
"""ostrack"""
trackers.extend(trackerlist(name='ostrack', parameter_name='baseline_fullx4_384', dataset_name=dataset_name,
                            run_ids=None, display_name='OSTrack256'))
# trackers.extend(trackerlist(name='lasot_450', parameter_name='baseline_fullx4_384_299', dataset_name=dataset_name,
#                             run_ids=None, display_name='OSTrack256'))
# trackers.extend(trackerlist(name='lasot_450', parameter_name='baseline_fullx4_384_298', dataset_name=dataset_name,
#                             run_ids=None, display_name='OSTrack256'))
# trackers.extend(trackerlist(name='lasot_450', parameter_name='baseline_fullx4_384_297', dataset_name=dataset_name,
#                             run_ids=None, display_name='OSTrack256'))
# trackers.extend(trackerlist(name='lasot_450', parameter_name='baseline_fullx4_384_296', dataset_name=dataset_name,
#                             run_ids=None, display_name='OSTrack256'))
# trackers.extend(trackerlist(name='lasot_450', parameter_name='baseline_fullx4_384_280', dataset_name=dataset_name,
#                             run_ids=None, display_name='OSTrack256'))
# trackers.extend(trackerlist(name='lasot_450', parameter_name='baseline_fullx4_384_240', dataset_name=dataset_name,
#                             run_ids=None, display_name='OSTrack256'))
# trackers.extend(trackerlist(name='lasot_text', parameter_name='baseline_fullx4_384_149', dataset_name=dataset_name,
#                             run_ids=None, display_name='OSTrack256'))
# trackers.extend(trackerlist(name='lasot_text', parameter_name='baseline_fullx4_384_148', dataset_name=dataset_name,
#                             run_ids=None, display_name='OSTrack256'))
# trackers.extend(trackerlist(name='lasot_text', parameter_name='baseline_fullx4_384_147', dataset_name=dataset_name,
#                             run_ids=None, display_name='OSTrack256'))
# trackers.extend(trackerlist(name='lasot_text', parameter_name='baseline_fullx4_384_146', dataset_name=dataset_name,
#                             run_ids=None, display_name='OSTrack256'))
# trackers.extend(trackerlist(name='uav', parameter_name='baseline_fullx4_384_150', dataset_name=dataset_name,
#                             run_ids=None, display_name='OSTrack256'))
# trackers.extend(trackerlist(name='uav', parameter_name='baseline_fullx4_384_149', dataset_name=dataset_name,
#                             run_ids=None, display_name='OSTrack256'))
# trackers.extend(trackerlist(name='uav', parameter_name='baseline_fullx4_384_148', dataset_name=dataset_name,
#                             run_ids=None, display_name='OSTrack256'))
# trackers.extend(trackerlist(name='uav', parameter_name='baseline_fullx4_384_147', dataset_name=dataset_name,
#                             run_ids=None, display_name='OSTrack256'))
# trackers.extend(trackerlist(name='uav', parameter_name='baseline_fullx4_384_146', dataset_name=dataset_name,
#                             run_ids=None, display_name='OSTrack256'))
# trackers.extend(trackerlist(name='ostrack', parameter_name='baseline_fullx1_99', dataset_name=dataset_name,
#                             run_ids=None, display_name='OSTrack256'))
# trackers.extend(trackerlist(name='ostrack', parameter_name='baseline_full_98', dataset_name=dataset_name,
#                             run_ids=None, display_name='OSTrack256'))
# trackers.extend(trackerlist(name='ostrack', parameter_name='baseline_full_97', dataset_name=dataset_name,
#                             run_ids=None, display_name='OSTrack256'))
# trackers.extend(trackerlist(name='ostrack_wo', parameter_name='baseline_full_96', dataset_name=dataset_name,
#                             run_ids=None, display_name='OSTrack256'))
# trackers.extend(trackerlist(name='ostrack_last', parameter_name='vitb_384_mae_32x4_ep300_298', dataset_name=dataset_name,
#                             run_ids=None, display_name='OSTrack256'))
# trackers.extend(trackerlist(name='ostrack_last', parameter_name='vitb_384_mae_32x4_ep300_297', dataset_name=dataset_name,
#                             run_ids=None, display_name='OSTrack256'))
# trackers.extend(trackerlist(name='ostrack_last', parameter_name='vitb_384_mae_32x4_ep300_296', dataset_name=dataset_name,
#                             run_ids=None, display_name='OSTrack256'))
# trackers.extend(trackerlist(name='ostrack', parameter_name='vitb_256_mae_32x4_ep300_300', dataset_name=dataset_name,
#                             run_ids=None, display_name='OSTrack256'))


dataset = get_dataset(dataset_name)
# dataset = get_dataset('otb', 'nfs', 'uav', 'tc128ce')
# plot_results(trackers, dataset, 'OTB2015', merge_results=True, plot_types=('success', 'norm_prec'),
#              skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05)
print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'norm_prec', 'prec'))
# print_results(trackers, dataset, 'UNO', merge_results=True, plot_types=('success', 'prec'))
