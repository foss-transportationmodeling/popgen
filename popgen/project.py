import logging
import os
import time
import yaml

from config import Config
from data import DB
from ipf import Run_IPF
from reweighting import Run_Reweighting, Reweighting_DS
from draw import Draw_Population
from output import Syn_Population


class Project(object):
    """This is the primary class to setup and run PopGen projects.


    """
    def __init__(self, config_loc):
        self.config_loc = config_loc

    def load_project(self):
        self._load_config()
        self._populate_project_properties()
        self._load_data()

    def _load_config(self):
        # TODO: validating config file for YAML
        # TODO: validating YAML config file for field types
        # TODO: validating YAML for consistency across fields/config elements
        config_f = file(self.config_loc, "r")
        config_dict = yaml.load(config_f)
        self._config = Config(config_dict)
        self.column_names_config = self._config.project.inputs.column_names
        self.entities = self._config.project.inputs.entities
        self.housing_entities = self._config.project.inputs.housing_entities
        self.person_entities = self._config.project.inputs.person_entities

    def _populate_project_properties(self):
        self.name = self._config.project.name
        self.location = os.path.abspath(self._config.project.location)

    def _load_data(self):
        self.db = DB(self._config)
        self.db.load_data()

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
        self._get_geo_ids()
        self._run_ipf()
        self._run_weighting()
        self._draw_sample()
        self._report_results()

    def _get_geo_ids(self):
        self.db.enumerate_geo_ids_for_scenario(self.scenario_config)

    def _run_ipf(self):
        self.run_ipf_obj = Run_IPF(self.entities,
                                   self.housing_entities,
                                   self.column_names_config,
                                   self.scenario_config, self.db)
        self.run_ipf_obj.run_ipf()
        print "IPF completed in: %.4f" % (time.time() - self.t)

    def _run_weighting(self):
        reweighting_config = self.scenario_config.parameters.reweighting
        # if reweighting_config.procedure == "ipu":
        #     self._run_ipu()
        # def _run_ipu(self):
        self.run_reweighting_obj = Run_Reweighting(
            self.entities, self.column_names_config,
            self.scenario_config, self.db)
        self.run_reweighting_obj.create_ds()
        self.run_reweighting_obj.run_reweighting(
            self.run_ipf_obj.region_constraints,
            self.run_ipf_obj.geo_constraints)
        print "Reweighting completed in: %.4f" % (time.time() - self.t)

    def _draw_sample(self):
        self.draw_population_obj = Draw_Population(
            self.scenario_config, self.db.geo_ids,
            self.run_reweighting_obj.geo_row_idx,
            self.run_ipf_obj.geo_frequencies,
            self.run_ipf_obj.geo_constraints,
            self.run_reweighting_obj.geo_stacked,
            self.run_reweighting_obj.region_sample_weights)
        self.draw_population_obj.draw_population()
        print "Drawing completed in: %.4f" % (time.time() - self.t)

    def _report_results(self):
        self.syn_pop_obj = Syn_Population(
            self.location,
            self.db,
            self.column_names_config,
            self.scenario_config,
            self.run_ipf_obj,
            self.run_reweighting_obj,
            self.draw_population_obj,
            self.entities,
            self.housing_entities,
            self.person_entities)
        self.syn_pop_obj.add_records()
        self.syn_pop_obj.prepare_data()
        self.syn_pop_obj.export_outputs()
        print "Results completed in: %.4f" % (time.time() - self.t)


def popgen_run(project_config):
    logger = logging.getLgger()
    pass

if __name__ == "__main__":
    import time
    from config import Config
    from data import DB

    t = time.time()
    p_obj = Project("../tutorials/1_basic_popgen_setup/configuration.yaml")
    p_obj.load_project()
    p_obj.run_scenarios()
    print "Time it took: %.4f" % (time.time() - t)
