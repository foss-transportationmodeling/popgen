import os
import time

import pandas as pd
import numpy as np


class Syn_Population(object):
    def __init__(
            self, location, db, column_names_config, scenario_config,
            run_ipf_obj, run_ipu_obj, draw_population_obj,
            entities, housing_entities, person_entities):

        self.location = location
        self.db = db
        self.column_names_config = column_names_config
        self.scenario_config = scenario_config

        self.run_ipf_obj = run_ipf_obj
        self.geo_constraints = run_ipf_obj.geo_constraints
        self.geo_frequencies = run_ipf_obj.geo_frequencies
        self.region_constraints = run_ipf_obj.region_constraints

        self.run_ipu_obj = run_ipu_obj
        self.geo_row_idx = run_ipu_obj.geo_row_idx
        self.geo_stacked = run_ipu_obj.geo_stacked
        self.region_sample_weights = run_ipu_obj.region_sample_weights

        self.draw_population_obj = draw_population_obj

        self.entities = entities
        self.housing_entities = housing_entities
        self.person_entities = person_entities

        self.geo_name = self.column_names_config.geo
        self.region_name = self.column_names_config.region
        self.hid_name = self.column_names_config.hid
        self.pid_name = self.column_names_config.pid
        self.unique_id_in_geo_name = "unique_id_in_geo"

        self.pop_syn = None
        self.pop_syn_data = {}

        # The following sort column variables
        # are useful to separate synthetic population when same
        # sample unit is drawn multiple times
        self.pop_syn_geo_id_columns = [self.geo_name,
                                       self.unique_id_in_geo_name]
        self.pop_syn_all_id_columns = [self.geo_name, self.hid_name,
                                       self.unique_id_in_geo_name]
        self.pop_syn_housing_matching_id_columns = [
            self.geo_name, self.hid_name]
        self.pop_syn_person_matching_id_columns = [
            self.geo_name, self.hid_name, self.pid_name]

        self.pop_rows_syn_dict = {}
        self.housing_syn_dict = {}
        self.person_syn_dict = {}
        self.controls = {}
        self.geo_controls = {}
        self.region_controls = {}

        self._create_preliminaries()

    def _create_preliminaries(self):
        self._create_ds()
        self._create_meta_data()
        self._create_prepare_output_directory()

    def _create_ds(self):
        self.housing_stacked_sample = self._get_stacked_sample(
            self.housing_entities)
        self.person_stacked_sample = self._get_stacked_sample(
            self.person_entities)

    def _create_meta_data(self):
        region_controls_config = self.scenario_config.control_variables.region
        geo_controls_config = self.scenario_config.control_variables.geo

        controls_config_list = [geo_controls_config, region_controls_config]
        for entity in self.entities:
            self.controls[entity] = self._return_controls_for_entity(
                controls_config_list, entity
                )

        controls_config_list = [geo_controls_config]
        for entity in self.entities:
            self.geo_controls[entity] = self._return_controls_for_entity(
                controls_config_list, entity
                )

        controls_config_list = [region_controls_config]
        for entity in self.entities:
            self.region_controls[entity] = self._return_controls_for_entity(
                controls_config_list, entity
                )

        self.entity_types_dict = {}
        for entity in self.housing_entities:
            self.entity_types_dict[entity] = "housing"
        for entity in self.person_entities:
            self.entity_types_dict[entity] = "person"

        self.entity_types = ["housing", "person"]

    def _create_prepare_output_directory(self):
        current_time_str = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
        foldername = ("%s %s" % (current_time_str,
                                 self.scenario_config.description))
        self.outputlocation = os.path.join(self.location,
                                           foldername)
        if not os.path.exists(self.outputlocation):
            os.makedirs(self.outputlocation)

        self.filetype_sep_dict = {"csv": ","}

    def _return_controls_for_entity(self, controls_config_list,
                                    entity):
        controls = []
        for controls_config in controls_config_list:
            controls += controls_config[entity].return_list()
        return controls

    def _get_stacked_sample(self, entities):
        sample_list = [self.db.sample[entity]
                       for entity in entities]
        stacked_sample = pd.concat(sample_list).fillna(0)
        stacked_sample.sort(inplace=True)
        return stacked_sample

    def add_records(self):
        geo_id_rows_syn_dict = self.draw_population_obj.geo_id_rows_syn_dict
        for geo_id, geo_id_rows_syn in geo_id_rows_syn_dict.iteritems():
            geo_id_pop_syn = (
                self._get_stacked_geo_for_geo_id(geo_id, geo_id_rows_syn))
            self.pop_rows_syn_dict[geo_id] = geo_id_pop_syn
            self.pop_rows_syn_dict[geo_id][self.unique_id_in_geo_name] = (
                range(1, geo_id_rows_syn.shape[0]+1))
            # print self.pop_rows_syn_dict[geo_id].head()
            # raw_input()

    def _add_records_for_geo_id(self, geo_id, geo_id_rows_syn):
        # if self.pop_syn is not None:
        #    self.pop_syn = self.pop_syn.append(
        #        self.get_records_for_geo_id(geo_id, geo_id_rows_syn),
        #        ignore_index=True)
        # else:
        #    self.pop_syn = self.get_records_for_geo_id(
        #    geo_id, geo_id_rows_syn)
        geo_id_pop_syn = (
            self._get_stacked_geo_for_geo_id(geo_id, geo_id_rows_syn))

        self.pop_rows_syn_dict[geo_id] = geo_id_pop_syn
        # self.housing_syn_dict[geo_id] = (
        #    self.get_housing_data_for_indexes(geo_id_pop_syn))
        # self.person_syn_dict[geo_id] = (
        #    self.get_person_data_for_indexes(geo_id_pop_syn))

    def _get_stacked_geo_for_geo_id(self, geo_id, geo_id_rows_syn):
        geo_id_pop_syn = self.geo_stacked.take(geo_id_rows_syn).copy()
        geo_id_pop_syn[self.geo_name] = geo_id
        # print "\trows", geo_id_pop_syn.shape
        return geo_id_pop_syn

    def _get_housing_data_for_indexes(self, geo_id_pop_syn):
        housing_data = (
            geo_id_pop_syn.loc[:, [self.geo_name]].join(
                self.housing_stacked_sample))
        print "housing rows:", housing_data.shape
        return housing_data

    def _get_person_data_for_indexes(self, geo_id_pop_syn):
        person_data = (
            geo_id_pop_syn.loc[:, [self.geo_name]].join(
                self.person_stacked_sample))
        print "person rows:", person_data.shape
        return person_data

    def prepare_data(self):
        self._stack_records()
        self._create_synthetic_population()
        self._create_index()

    def _stack_records(self):
        t = time.time()
        self.pop_syn = pd.concat(
            self.pop_rows_syn_dict.values(), copy=False)
        # self.pop_syn_data["housing"] = pd.concat(
        #    self.housing_syn_dict.values(), copy=False)
        # self.pop_syn_data["person"] = pd.concat(
        #    self.person_syn_dict.values(), copy=False)
        # print self.pop_syn_data["housing"].shape
        # print self.pop_syn_data["person"].shape
        print "Time elapsed for stacking population is : %.4f" % (
            time.time() - t)

    def _create_synthetic_population(self):
        t = time.time()
        self.pop_syn_data["housing"] = (
            self.pop_syn.loc[:, self.pop_syn_geo_id_columns].join(
                self.housing_stacked_sample))
        self.pop_syn_data["person"] = (
            self.pop_syn.loc[:, self.pop_syn_geo_id_columns].join(
                self.person_stacked_sample))
        print "\tSize of the housing population table:", (
            self.pop_syn_data["housing"].shape)
        print "\tSize of the person population table:", (
            self.pop_syn_data["person"].shape)
        # print "Time elapsed for synthetic population 1 is : %.4f" % (
        #    time.time() - t)

    def _create_index(self):
        t = time.time()
        # self.pop_syn_data["housing"].reset_index(inplace=True)
        self.pop_syn_data["housing"].set_index(
            self.pop_syn_housing_matching_id_columns, inplace=True, drop=False)
        self.pop_syn_data["housing"].sort(inplace=True)

        # self.pop_syn_data["person"].reset_index(inplace=True)
        self.pop_syn_data["person"].set_index(
            self.pop_syn_person_matching_id_columns, inplace=True,
            drop=False)
        self.pop_syn_data["person"].sort(inplace=True)
        # print "Time elapsed for index is : %.4f" % (time.time() - t)

    def export_outputs(self):
        print "\tGenerating Outputs"
        t = time.time()
        self._export_performance_data()
        self._export_multiway_tables()
        self._export_summary()
        self._export_weights()
        self._export_synthetic_population()
        self._pretty_print_scenario_configuration_file_to_output()
        print "\tTime elapsed for generating outputs is : %.4f" % (
            time.time() - t)

    def _pretty_print_scenario_configuration_file_to_output(self):
        filepath = os.path.join(self.outputlocation,
                                "%s.yaml" % self.scenario_config.description)
        self.scenario_config.write_to_file(filepath)

    def _export_performance_data(self):
        values_to_export = self.scenario_config.outputs.performance
        # print "Performance values to export:", values_to_export
        if "ipf" in values_to_export:
            self._export_all_df_in_dict(
                self.run_ipf_obj.geo_iters_convergence_dict,
                "ipf_geo_iters_convergence_")
            self._export_all_df_in_dict(
                self.run_ipf_obj.geo_average_diffs_dict,
                "ipf_geo_average_diffs_")
            self._export_all_df_in_dict(
                self.run_ipf_obj.region_iters_convergence_dict,
                "ipf_region_iters_convergence_")
            self._export_all_df_in_dict(
                self.run_ipf_obj.region_average_diffs_dict,
                "ipf_region_average_diffs_")
        if "reweighting" in values_to_export:
            self._export_df(
                self.run_ipu_obj.average_diffs,
                "reweighting_average_diffs")
        if "drawing" in values_to_export:
            self._export_df(
                self.draw_population_obj.draws_performance, "draws")

    def _export_weights(self):
        export_weights_config = self.scenario_config.outputs.weights
        if export_weights_config.export:
            df = pd.DataFrame(self.run_ipu_obj.region_sample_weights)
            if export_weights_config.collate_across_geos:
                df = df.sum(axis=1)
            self._export_df(df, "weights")

    def _export_df(self, df, filename):
        filepath = os.path.join(self.outputlocation, "%s.csv" % filename)
        df.to_csv(filepath)

    def _export_all_df_in_dict(self, dict_of_dfs, fileprefix):
        for key, value in dict_of_dfs.iteritems():
            filename = "%s%s.csv" % (fileprefix, key)
            filepath = os.path.join(self.outputlocation, filename)
            value.to_csv(filepath)

    def _export_multiway_tables(self):
        multiway_tables = self._return_multiway_tables()

        for (filename, filetype), table in multiway_tables.iteritems():
            filepath = os.path.join(self.outputlocation, filename)
            table.to_csv(
                filepath, sep=self.filetype_sep_dict[filetype])

    def _return_multiway_tables(self):
        multiway_tables = {}
        for table_config in self.scenario_config.outputs.multiway:
            t = time.time()
            (variables, filename, filetype,
             entity) = (
                table_config.variables.return_list(), table_config.filename,
                table_config.filetype, table_config.entity)
            entity_type = self.entity_types_dict[entity]
            multiway_table_entity = self._return_aggregate_by_geo(
                variables, entity_type, entity)
            multiway_tables[(filename, filetype)] = multiway_table_entity
            print "\t\tTime elapsed for each table is: %.4f" % (
                time.time() - t)
        return multiway_tables

    def _export_synthetic_population(self):
        t = time.time()
        synthetic_population_config = (
            self.scenario_config.outputs.synthetic_population)
        sort_columns = self.pop_syn_all_id_columns
        for entity_type in self.entity_types:
            (filename, filetype) = (
                synthetic_population_config[entity_type].filename,
                synthetic_population_config[entity_type].filetype)
            filepath = os.path.join(self.outputlocation, filename)
            # self.pop_syn_data[entity_type].to_csv(
            #    filepath, sep=self.filetype_sep_dict[filetype], index=False)
            self.pop_syn_data[entity_type].sort(
                sort_columns, inplace=True)
            self.pop_syn_data[entity_type].reset_index(drop=True, inplace=True)
            self.pop_syn_data[entity_type].index.name = (
                "unique_%s_id" % entity_type)
            self.pop_syn_data[entity_type].to_csv(
                filepath, sep=self.filetype_sep_dict[filetype])
        print "\tTime to write syn pop files is: %.4f" % (time.time() - t)

    def _return_aggregate_by_geo(self, variables, entity_type, entity):
        if isinstance(variables, str):
            variables = [variables]
        groupby_columns = ["entity", self.geo_name] + variables
        columns_count = len(groupby_columns)
        multiway_table = (self.pop_syn_data[entity_type].groupby(
            groupby_columns).size())
        multiway_table_entity = (
            multiway_table[entity].unstack(level=range(1, columns_count-1))
            )
        return multiway_table_entity

    def _export_summary(self):
        t = time.time()
        summary_config = self.scenario_config.outputs.summary
        marginal_geo = self._return_marginal_geo()
        (geo_filename, geo_filetype) = (
            summary_config.geo.filename, summary_config.geo.filetype)
        filepath = os.path.join(self.outputlocation, geo_filename)
        marginal_geo.to_csv(
            filepath, sep=self.filetype_sep_dict[geo_filetype])
        # print marginal_geo

        marginal_region = self._return_marginal_region(marginal_geo)
        (region_filename, region_filetype) = (
            summary_config.region.filename, summary_config.region.filetype)
        filepath = os.path.join(self.outputlocation, region_filename)
        marginal_region.to_csv(
            filepath, sep=self.filetype_sep_dict[region_filetype])
        print "\tSummary creation took: %.4f" % (time.time() - t)
        # print marginal_region

    def _return_marginal_region(self, marginal_geo):
        region_to_geo = self.db.geo["region_to_geo"]
        marginal_region = region_to_geo.join(
            marginal_geo, on=self.geo_name, how="inner")[marginal_geo.columns]
        marginal_region = marginal_region.reset_index().groupby(
            self.region_name).sum()

        marginal_region.columns = pd.MultiIndex.from_tuples(
            marginal_region.columns)
        return marginal_region

    def _return_marginal_geo(self):
        marginal_list = []
        for entity in self.entities:
            entity_type = self.entity_types_dict[entity]
            for variable in self.controls[entity]:
                variable_marginal = (
                    self._return_aggregate_by_geo(
                        variable, entity_type, entity))
                marginal_list.append(
                    variable_marginal)
        marginal_geo = self._stack_marginal(marginal_list)
        return marginal_geo

    def _stack_marginal(self, marginal_list):
        marginal_T_list = []
        for marginal in marginal_list:
            marginal = marginal.T.copy()
            marginal["name"] = marginal.index.name
            marginal_T_list.append(marginal)
        stacked_marginal = pd.concat(marginal_T_list)
        stacked_marginal.index.name = "categories"
        stacked_marginal.reset_index(inplace=True)
        stacked_marginal.set_index(["name", "categories"], inplace=True)
        stacked_marginal.sort(inplace=True)  # Sort by row indices
        return stacked_marginal.T

    def _report_summary(self, geo_id_rows_syn, geo_id_frequencies,
                        geo_id_constraints, over_columns=None):
        geo_id_synthetic = self.geo_stacked.take(geo_id_rows_syn).sum()
        geo_id_synthetic = pd.DataFrame(geo_id_synthetic,
                                        columns=["synthetic_count"])
        geo_id_synthetic["frequency"] = geo_id_frequencies
        geo_id_synthetic["constraint"] = geo_id_constraints
        geo_id_synthetic["diff_constraint"] = (
            geo_id_synthetic["synthetic_count"] -
            geo_id_synthetic["constraint"])
        geo_id_synthetic["abs_diff_constraint"] = (
            geo_id_synthetic["diff_constraint"].abs())
        geo_id_synthetic["diff_frequency"] = (
            geo_id_synthetic["synthetic_count"] -
            geo_id_synthetic["frequency"])
        geo_id_synthetic["abs_diff_frequency"] = (
            geo_id_synthetic["diff_frequency"].abs())

        stat, p_value = stats.chisquare(geo_id_synthetic["synthetic_count"],
                                        geo_id_synthetic["constraint"])
        aad_in_frequencies = (geo_id_synthetic["abs_diff_frequency"]).mean()
        aad_in_constraints = (geo_id_synthetic["abs_diff_constraint"]).mean()
        sad_in_constraints = (geo_id_synthetic["abs_diff_constraint"]).sum()
        sd_in_constraints = (geo_id_synthetic["diff_constraint"]).sum()

        print "%.4f, %f, %f, %f, %f, %f" % (stat, p_value, aad_in_frequencies,
                                            aad_in_constraints,
                                            sad_in_constraints,
                                            sd_in_constraints)
