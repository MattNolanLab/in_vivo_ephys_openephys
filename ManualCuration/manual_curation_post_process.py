import ManualCuration.manual_curation_settings


def post_process_manually_curated_data(recording_server, recording_local):
    pass
    # read phy output and split firing times back and save as spatial_firing_curated (also for paired recordings)
    # save output back on server (copy manual spatial firing back)
    ##do after this: change pipeline so it loads manual spatial firing if it exists (?)


def main():
    recording_server = ManualCuration.manual_curation_settings.get_recording_path_datastore()
    recording_local = ManualCuration.manual_curation_settings.get_local_recording_path()
    print('This script will process the manually curated phy outputs and upload them to datastore: ' + recording_server)
    post_process_manually_curated_data(recording_server, recording_local)


if __name__ == '__main__':
    main()