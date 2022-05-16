## CO3_CONFIG_PATH

The environment variable, CO3_CONFIG_PATH, contains a list of paths
where co3 configuration files are to be found. Each such path is separated
with a ":" character and in this sense, it is similar to PYTHONPATH. It is commonly
referred to when loading the configuration for both playbacks and
generates.

## ./co3-configurations/dev

This folder now contains a set of demonstration YAML files which, taken together, are
meant to demonstrate typical configurations. These files should be
referred to as further configurations are developed and historical ones
are modified to accommodate this release.

### co3.yaml

Noteworthy are the following:

    console: can be set to logging or progress_bar; if not set, it
        defaults to logging.

    defaults_ : used to identify a default configuration file (this
        file itself is discussed below). Note the underscore in the key
        name, which is required in order to distinguish from Hydra's use
        of 'defaults'.

    playbacks: A collection of one or more playbacks.

        <playback_name>

            agent:
<div style="padding-left: 120px;">
Parameters specific to the agent function. Many parameters are common to all agents and hence it is
these that can be stored in the defaults_ file. The common set of parameters are discussed here.
Agent-specific parameters are discussed in the section Agent-Specific Parameters.
</div>

                action_noise:

                    type:
                    To be selected from null | OrnsteinUhlenbeck | Gaussian

                    sigma:
                    Standard deviation of the noise source.

                batch_size:
                    Size of the batch to be sampled from experience
                    replay buffer during training.

                buffer_size:
                    Size of the experiences replay buffer. batch_size
                    :<= buffer_size

                - episodes:
                    The number of episodes that the process is to run.

                -   gamma:
                    Bellman discount factor

                -   [gradient_clipping](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html):
                    Further details available at:
                    <https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html>

                    -   max_norm

                    -   norm_type

                -   purge_network:
                    Boolean defining whether the network is to be purged
                    before the run begins.

                -   network:
                    Defines a path name or a file name of a PyTorch
                    network (extension: :*.pt). If this parameter is not
                    provided, then a new network file will be created in
                    the location \`CO3_PATH/networks/\` (called the
                    \'nominal location\' below) with file name
                    \`\<current datetime\>.pt\`. If this parameter is a
                    path name and it already exists, then it must have
                    content that is recognizable as a valid network file
                    by PyTorch. If the path name does not currently
                    exist, then a new network file will be created. On
                    the other hand, if a file name is provided, then the
                    software will check to see if the file already
                    exists in the nominal location and, if so, it will
                    expect that this file is one that is recognizable as
                    a network file by PyTorch. If no such file is found,
                    then a new file of that name will be created in the
                    nominal location.

                -   training\
                    Boolean selecting whether the network is to be
                    trained during the course of the run.

                -   training_interval\
                    Number of steps between trainings of the network.

                -   target_update_interval\
                    Number of trainings between updating the target

            -   misc: Parameters that are process-general and not
                associated with a specific function such as the agent or
                environment. These are discussed now:

                -   csv_path\
                    Path to where the rewards file is to be recorded. If
                    null, then the path name is automatically generated.
                    If defined and a relative path, then the pathname
                    must begin with 'rewards'.

                -   generate_csv\
                    A boolean variable defining whether a rewards csv is
                    generated or not.

                -   leave_progress_bar\
                    If set, then completed (test) progress bars are
                    retained; otherwise complete test progress bars are
                    discared.

                -   log_interval\
                    The number of episodes between routine logging
                    messages.

                -   log_level\
                    The logging modules log level. To be selected from
                    \[DEBUG \| INFO \| WARNING \| ERROR \| CRITICAL\].
                    See <https://docs.python.org/3/library/logging.html>
                    for further details.

                -   seed\
                    Seed to be used for all random number generators
                    accessed by the software (Python, numpy, OpenAI gym,
                    and PyTorch).

                -   record_state\
                    Record (or not) state in the rewards CSV file.

            -   env_config

                -   envId

                -   exchange

                -   reward_offset

                -   randomize_dataset_reads

                -   pdf

                    -   age_limit\
                        Expressed in days, can be decimal; e.g. 0.25 = 6
                        hrs

                    -   bins\
                        Further details can be found here:
                        <https://numpy.org/doc/stable/reference/generated/numpy.histogram.html>

                    -   graph\
                        Boolean selecting whether to graph a newly
                        generated PDF.

            -   child_process\
                When a child process is created, its configuration
                parameters are a merge of the parents configuration
                parameters with the parameters belonging to the
                child_process key, although the merge has a number of
                limitations as follows:

                -   A child process itself cannot have a child_process
                    key.

                -   agent.training is hardwired to be False (child
                    process networks are never trained).

                -   Only the following parameters can be set to
                    different values from that of the parent (training)
                    process:

                    1.  misc.csv_path

                    2.  misc.generate_csv

                    3.  misc.log_interval

                    4.  agent.episodes

                    5.  agent.nn_trace

                    6.  env_config.datasets

                    7.  env_config.randomize_dataset_reads
