ContikiPy - Automated Cooja Simulations, Log Parsing, and Plotting with Python.
==
### Intro
This repo hosts the source code for ContikiPy, a suite of Python scripts for
automated simulation running in cooja, as well as log parsing and plotting.

### About
ContikiPy is a pet project that began as an attempt to try and make the task of getting information and results out of the Contiki logs a bit easier. There has been no formal design as such, just a bunch of things that seemed like a good idea to try and do.

Current YAML configurations are set up for my own experiments (based on Contiki-NG logging), but they should provide a good template for making your own.

### Development
I work with Contiki a fair amount so I'm constantly trying to improve usability, but if you have a pull request / any suggestions then feel free to contribute!

### Getting Started

#### Running ContikiPy
You can run ContikiPy through the following command, where you replace *"my-config-file-xx"* with your own YAML configuration:

- *./contikipy.py --conf="my-config-file-xx.yaml" --runcooja=1 --parse=1 --comp=1*

A full list of commands is available through:

- *./contikipy.py --help*

#### Options
The three main options are as follows:
- runcooja - *run simulations according to the YAML config*
- parse - *parse simulation logs according to the YAML config and output plots*
- comp - *compare plots according to the YAML config*


#### Adding New Plots
1. In the yaml config:
 - Add the name of your plot to 'plot:' in your yaml simulation.
2. In cplogparser.py:
 - Add your plot to 'atomic_function_map' in plot_data() in cplogparser.py.
 - Map your plot name to the plot function name.
 - Add your plot to 'atomic_dict_map' in plot_data in cplogparser.
 - Map your plot name to the required data sets.
 - Write your plot function. It will take in a list of pandas dataframes
  (that you have previously mapped to this function) which you can use for
  generating your plot.
