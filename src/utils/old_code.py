def get_file_name(file_info: Dict[str, str]) -> str:
    """
    Returns the file name for storing the results or scores based on the file information.

    :param file_info: The information about the file to store.
    :return: The file name for storing the results or scores.
    """
    file_name = (f"{file_info['experiment_name']}_{file_info['model_alias']}_"
                 f"{file_info['num_articles']}_{file_info['min_new_tokens']}_"
                 f"{file_info['max_new_tokens']}_{file_info['num_beams']}")
    return file_name

def store_results_or_scores(file: dict, file_info: Dict[str, str]) -> None:
    """
    Stores the results (generated summaries) or scores (topic and quality scores
    for generated summaries) in the filesystem.

    :param file: The file to store.
    :param file_information: The information about the file to store.
    """
    file_name = f"{file_info['output_type']}_{get_file_name(file_info)}.json"
    data_path = utils.get_path("DATA_PATH")
    if not data_path:
        logging.error(f"Data directory not found. "
                        f"{file_info['output_type'].capitalize} have not been stored.")
        return

    results_dir = os.path.join(data_path, file_info['output_type'],
                                f"{file_info['experiment_name']}", file_name)
    if file_info['output_type'] in ['results', 'scores']:
        with open(results_dir, 'w', encoding='utf-8') as f:
            json.dump(file, f, indent=4)
    else:
        logger.error(f"Invalid output type: {file_info['output_type']}. "
                     "Results or scores have not been stored.")

    logging.info(f"{file_info['output_type'].capitalize()} successfully stored.")
    return
