import logging
import os
import time

from config import Config
from data import DB
from ipf import Run_IPF
from reweighting import Run_IPU, Reweighting_DS
from draw import Draw_Population
from output import Syn_Population


class Project(object):
    """This is the primary class to setup and run PopGen projects. This accepts
    a configuration file which is in a YAML format.
    """
    def __init__(self, config_loc):
        self._config_loc = config_loc

    def load_project(self):
        self._load_config()
        self._populate_project_properties()
        self._load_data()
        self._enumerate_geos()

    def _load_config(self):
        #TODO: validating config file for YAML
        #TODO: validating YAML config file for field types
        #TODO: validating YAML for consistency across fields/config elements
        config_f = file(self._config_loc, "r")
        config_dict = yaml.load(config_f)
        self._config = Config(config_dict)
        self.column_names_config = self._config.project.inputs.column_names
        self.entities = self._config.project.inputs.entities
        self.housing_entities = self._config.project.inputs.housing_entities
        self.person_entities = self._config.project.inputs.person_entities

    def _populate_project_properties(self):
        self.name = self._config.project.name
        self.synthesize = self._config.project.synthesize
        self.location = os.path.abspath(self._config.project.location)

    def _load_data(self):
        self.db = DB(self._config)
        self.db.load_data()

    def _enumerate_geos(self):
        self.db.enumerate_geo_ids()

    def run_scenarios(self):
        scenarios_config = self._config.project.scenario
        for scenario_config in scenarios_config:
            print "Running Scenario: %s" % scenario_config.description
            scenario_obj = Scenario(self.location,
                                    self.entities, self.housing_entities,
                                    self.person_entities,
                                    self.column_names_config,
                                    scenario_config, self.db)
            scenario_obj.run_scenario()


class Scenario(object):
    def __init__(self, location, entities, housing_entities, person_entities,
                 column_names_config, scenario_config, db):
        self.location = location
        self.entities = entities
        self.housing_entities = housing_entities
        self.person_entities = person_entities
        self.column_names_config = column_names_config
        self.scenario_config = scenario_config
        self.db = db
        self.t = time.time()

    def run_scenario(self):
        self._run_ipf()
        self._run_weighting()
        self._draw_sample()
        self._report_results()

    def _run_ipf(self):
        self.run_ipf_obj = Run_IPF(self.entities,
                                   self.housing_entities,
                                   self.column_names_config,
                                   self.scenario_config, self.db)
        self.run_ipf_obj.run_ipf()
        print "IPF completed in: %.4f" % (time.time() - self.t)

    def _run_weighting(self):
        reweighting_config = self.scenario_config.parameters.reweighting
        if reweighting_config.procedure == "ipu":
            self._run_ipu()
        print "Reweighting completed in: %.4f" % (time.time() - self.t)

    def _run_ipu(self):
        self.run_ipu_obj = Run_IPU(self.entities, self.column_names_config,
                                   self.scenario_config, self.db)
        self.run_ipu_obj.create_ds()
        self.run_ipu_obj.run_ipu(self.run_ipf_obj.region_constraints,
                                 self.run_ipf_obj.geo_constraints)

    def _draw_sample(self):
        self.syn_pop_obj = Syn_Population(
            self.location,
            self.db,
            self.column_names_config,
            self.scenario_config,
            self.run_ipf_obj.geo_constraints,
            self.run_ipf_obj.geo_frequencies,
            self.run_ipf_obj.region_constraints,
            self.run_ipu_obj.geo_row_idx,
            self.run_ipu_obj.geo_stacked,
            self.run_ipu_obj.region_sample_weights,
            self.entities,
            self.housing_entities,
            self.person_entities)

        self.draw_population_obj = Draw_Population(self.scenario_config,
                                                   self.syn_pop_obj)
        self.draw_population_obj.draw_population()
        print "Drawing completed in: %.4f" % (time.time() - self.t)

    def _report_results(self):
        self.syn_pop_obj.prepare_data()
        self.syn_pop_obj.export_outputs()
        print "Results completed in: %.4f" % (time.time() - self.t)


def popgen_run(project_config):
    logger = logging.getLgger()
    pass

if __name__ == "__main__":
    import yaml
    import time
    from config import Config
    from data import DB

    t = time.time()
    p_obj = Project("../demo/bmc_taz/configuration.yaml")
    p_obj.load_project()
    p_obj.run_scenarios()
    print "Time it took: %.4f" % (time.time() - t)
