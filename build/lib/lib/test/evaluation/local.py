from lib.test.evaluation.environment import EnvSettings
import vot
def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = ''
    settings.got10k_path = ''
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_lmdb_path = ''
    settings.lasot_path = '/media/liyunfeng/CV2/data/sot/lasot'
    settings.network_path = '/home/liyunfeng/code/project2/Stark/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = '/media/liyunfeng/CV2/data/sot/otb'
    settings.prj_dir = '/home/liyunfeng/code/project2/Stark'
    settings.result_plot_path = '/home/liyunfeng/code/project2/Stark/test/result_plots'
    settings.results_path = '/home/liyunfeng/code/project2/Stark/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/liyunfeng/code/project2/Stark'
    settings.segmentation_path = '/home/liyunfeng/code/project2/Stark/test/segmentation_results'
    settings.tc128_path =''
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.uav_path = ''
    settings.vot_path = ''
    settings.youtubevos_dir = ''

    return settings

