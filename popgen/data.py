import os

import pandas as pd
import numpy as np


class DB(object):
    """This class returns a Inputs object that can be used to handle all
    the Popgen input data files. All input files are stored as Pandas
    dataframe objects. The object maintains the following PopGen inputs:
    - Two geographic correspondence mapping inputs
    1) a mapping between region and geo
    2) a mapping between geo and sample_geo
    - Two sets of sample data files
    1) for housing entities
    2) for person entities
    - Two sets of marginal data files
    1) for housing entities
    2) for person entities
    """
    def __init__(self, config):
        self.config = config
        self.sample = {}
        self.geo_marginals = {}
        self.region_marginals = {}
        self.geo = {}
        self.geo_ids = None
        self.region_ids = None
        self.sample_geo_ids = None
        self._inputs_config = self.config.project.inputs

    def load_data(self):

        geo_corr_mapping_config = self._inputs_config.location.geo_corr_mapping
        self.geo = self.get_data(geo_corr_mapping_config)

        sample_config = self._inputs_config.location.sample
        self.sample = self.get_data(sample_config)

        geo_marginals_config = self._inputs_config.location.marginals.geo
        self.geo_marginals = self.get_data(geo_marginals_config,
                                           header=[0, 1])

        region_marginals_config = self._inputs_config.location.marginals.region
        self.region_marginals = self.get_data(region_marginals_config,
                                              header=[0, 1])

    def get_data(self, config, header=0):
        config_dict = config.return_dict()
        #print config_dict, type(config_dict)
        data_dict = {}
        for item in config_dict:
            full_location = os.path.abspath(config_dict[item])
            data_dict[item] = pd.read_csv(full_location, index_col=0,
                                          header=header)
            data_dict[item].loc[:,
                                data_dict[item].index.name] = (data_dict[item]
                                                               .index.values)
            #print data_dict[item]
        #print data_dict.keys()
        return data_dict

    def enumerate_geo_ids(self):
        geo_to_sample = self.geo["geo_to_sample"]
        self.geo_ids = geo_to_sample.index.values
        self.sample_geo_ids = np.unique(geo_to_sample[self._inputs_config
                                                      .column_names
                                                      .sample_geo].values)

        region_to_geo = self.geo["region_to_geo"]
        self.region_ids = np.unique(region_to_geo.index.values)

        #region_to_sample = self.geo["region_to_sample"]
        #self.region_ids = np.unique(region_to_sample.index.values)

    def get_geo_ids_for_region(self, region_id):
        geo_name = self._inputs_config.column_names.geo
        return self.geo["region_to_geo"].loc[region_id, geo_name].copy()

    def enumerate_geo_ids_to_synthesize(self):
        #TODO: Implement this to only synthesize a few geographies
        pass

    def return_variables_cats(self, entity, variable_names):
        variables_cats = {}
        for variable_name in variable_names:
            variables_cats[variable_name] = (self.return_variable_cats
                                             (entity, variable_name))
        return variables_cats

    def return_variable_cats(self, entity, variable_name):
        return np.unique(self.sample[entity][variable_name].values).tolist()

    def check_data(self):
        self.check_sample_marginals_consistency()
        self.check_marginals()

    def check_sample_margianls_consistency(self):
        #TODO: check consistency in variables across files
        #TODO: check consistency in categories across files
        pass

    def check_marginals(self):
        #TODO: check consistency in marginals across
        #TODO: check geo ids, sample geo ids, region ids across files
        pass
