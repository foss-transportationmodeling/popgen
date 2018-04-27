import numpy as np
import pandas as pd
import time
from scipy.optimize import fsolve, newton
from config import ConfigError


# TODO: Move all DS processing to Syn_Population Class
class Reweighting_DS(object):
    def __init__(self, ds_format="full"):
        self.ds_format = ds_format

    def get_sample_restructure1(self,
            entity, sample, variable_names, hid_name, sample_geo_names=None,
            adjust_household_ipf_config=None):
        sample["entity"] = entity

        groupby_columns = [hid_name, "entity"] + variable_names
        columns_count = len(groupby_columns)
        unstack_range = range(1, columns_count)
        sample_restruct = (sample.groupby(groupby_columns)
                           .size()
                           .unstack(level=unstack_range)
                           .fillna(0)
                           )

        if adjust_household_ipf_config is not None:
            groupby_columns = [hid_name, "entity"]
            column_index = pd.MultiIndex(levels=[
                [adjust_household_ipf_config.person_total_marginal_variable],
                [1]], labels=[[0], [0]])
            additional_information = pd.DataFrame(
                sample.groupby(groupby_columns).size(), columns=column_index)
            sample_restruct = sample_restruct.join(additional_information)

        if sample_geo_names is not None:
            sample_restruct = sample_restruct.join(
                sample[sample_geo_names])

        sample_restruct.columns = pd.Index(sample_restruct.columns,
                                           tuplelize_cols=False)
        sample_restruct = sample_restruct.astype(int)
        return sample_restruct

    def get_sample_restructure(self,
            entity, sample, variable_names, hid_name, sample_geo_names=None,
            adjust_household_ipf_config=None):
        sample["entity"] = entity
        additional_columns = []

        if adjust_household_ipf_config is None:
            groupby_columns = [hid_name, "entity"] + variable_names
            columns_count = len(groupby_columns)
            unstack_range = range(1, columns_count)
            sample_restruct = (sample.groupby(groupby_columns)
                               .size()
                               .unstack(level=unstack_range)
                               .fillna(0)
                               )

        if adjust_household_ipf_config is not None:
            other_columns = ["entity"] + variable_names
            values_for_other_columns = [
                tuple(row) for row in sample[other_columns].values]
            sample["composite_from_other_columns"] = \
                values_for_other_columns

            groupby_columns = [hid_name, "composite_from_other_columns"]

            columns_count = len(groupby_columns)
            unstack_range = range(1, columns_count)
            sample_restruct = (sample.groupby(groupby_columns)
                               .size()
                               .unstack(level=unstack_range)
                               .fillna(0)
                               )

            additional_columns += [
                "composite_from_other_columns",
                adjust_household_ipf_config.size_sample_variable[entity]]

        if sample_geo_names is not None:
            additional_columns += sample_geo_names

        if len(additional_columns) > 0:
            sample_restruct = sample_restruct.join(
                    sample[additional_columns])

        if adjust_household_ipf_config is not None:

            # The person dummy is hardcoded - needs to be revisited later
            # print sample_restruct.columns
            adjust_household_ipf_column_rename_dict = {
                adjust_household_ipf_config.size_sample_variable[entity]:
                    adjust_household_ipf_config.person_total_marginal_variable
                }

            sample_restruct.rename(
                columns=adjust_household_ipf_column_rename_dict,
                inplace=True)

        sample_restruct = sample_restruct.astype(int, errors="ignore")
        sample_restruct.columns = pd.Index(sample_restruct.columns,
                                           tuplelize_cols=False)
        # print sample_restruct.head()
        # print adjust_household_ipf_config
        # raw_input("Sample struct for entity: {0}".format(entity))

        return sample_restruct

    def get_stacked_sample_restruct(
            self, sample_restruct_list, sample_geo_names=None,
            adjust_household_ipf_config=None):

        # stacked_sample = pd.concat(sample_restruct_list, axis=1)
        if len(sample_restruct_list) == 1:
            return sample_restruct_list[0]

        common_cols = []

        if sample_geo_names is not None:
            common_cols += sample_geo_names

        if adjust_household_ipf_config is not None:
            common_cols += [
                "composite_from_other_columns",
                adjust_household_ipf_config.person_total_marginal_variable
                ]

        sample_restruct_common_cols_list = []
        for sample_restruct in sample_restruct_list:
            sample_restruct_common_cols_list.append(
                sample_restruct[common_cols])

        stacked_sample = pd.concat(
            sample_restruct_common_cols_list)

        for sample_restruct in sample_restruct_list:
            cols_to_merge = list(
                set(sample_restruct.columns.tolist()) - set(common_cols))
            stacked_sample = stacked_sample.join(sample_restruct[cols_to_merge])

        # print stacked_sample.columns

        stacked_sample.fillna(0., inplace=True)
        stacked_sample = stacked_sample.astype(int, errors="ignore")
        stacked_sample.sort_index(inplace=True)  # Sort by row indices
        stacked_sample.sort_index(axis=1,
                                  inplace=True)  # Sort columns alphabetically
        stacked_sample.columns = pd.Index(stacked_sample.columns,
                                          tuplelize_cols=False)

        # stacked_sample.groupby([
        #    "composite_from_other_columns",
        #    adjust_household_ipf_config.person_total_marginal_variable
        #    ]).sum()

        # print stacked_sample.head()
        # raw_input("Check stacked sample here")
        return stacked_sample

    def get_row_idx(self, sample_restruct):
        row_idx = {}
        contrib = {}
        for column in sample_restruct.columns.values.tolist():
            # print ("""Row IDX for column: {0} and dtype is: {1} """
            #       """and inferred dtype is: {2}""".format(
            #    column, sample_restruct[column].dtype,
            #    sample_restruct[column].infer_objects().dtype))
            rows = np.where(sample_restruct[column] > 0)[0]
            row_idx[column] = rows
            if self.ds_format == "full":
                contrib[column] = np.array(
                    sample_restruct[column].values, order="C",
                    dtype=sample_restruct[column].infer_objects().dtype)
            elif self.ds_format == "contributing":
                contrib[column] = np.array(
                    sample_restruct[column].values[rows], order="C",
                    dtype=sample_restruct[column].infer_objects().dtype)
            else:
                raise (Exception,
                       "Invalid value for config element ds_format")
        return (row_idx, contrib)

    def get_groupby(self,
            sample_restruct, sample_geo_names, adjust_household_ipf_config,
            hid_name=None):

        if hid_name is not None:
            return self._get_groupby_disaggregate(
                sample_restruct, sample_geo_names, adjust_household_ipf_config,
                hid_name)
        else:
            return self._get_groupby_aggregate(
                sample_restruct, sample_geo_names, adjust_household_ipf_config)

    def _get_groupby_aggregate(self, sample_restruct,
            sample_geo_names, adjust_household_ipf_config):

        groupby_columns = []

        groupby_columns += ["composite_from_other_columns",
            adjust_household_ipf_config.person_total_marginal_variable]

        sample_restruct_groupby = \
            sample_restruct.groupby(groupby_columns).max()
        sample_restruct_groupby = sample_restruct_groupby[
            sample_restruct_groupby.columns.difference(
                sample_geo_names)]

        sample_restruct_groupby[
            adjust_household_ipf_config.person_total_marginal_variable] = \
            sample_restruct_groupby.index.get_level_values(level=1)

        # print sample_restruct_groupby
        # raw_input("Check the _create_intermediate_sample_weights_df")
        return sample_restruct_groupby


    def _get_groupby_disaggregate(self, sample_restruct,
            sample_geo_names, adjust_household_ipf_config,
            hid_name):

        groupby_columns = []

        # print sample_restruct.head()

        groupby_columns += [hid_name, "composite_from_other_columns",
            adjust_household_ipf_config.person_total_marginal_variable]

        columns_count = len(groupby_columns)
        unstack_range = range(1, columns_count)

        sample_restruct_groupby = \
            (sample_restruct.groupby(groupby_columns)
             .size()
             .unstack(level=unstack_range)
            .fillna(0)
            )

        sample_restruct_groupby.columns = pd.Index(
            sample_restruct_groupby.columns,
            tuplelize_cols=False)

        # print sample_restruct_groupby.head()
        # raw_input("Check the _create_intermediate_sample_weights_df")
        return sample_restruct_groupby


class Run_Reweighting(object):
    def __init__(self, entities, housing_entities, person_entities,
                 column_names_config, reweighting_config,
                 control_variables_config, db,
                 adjust_household_ipf_config=None):
        self.entities = entities
        self.housing_entities = housing_entities
        self.person_entities = person_entities
        self.column_names_config = column_names_config
        self.hid_name = self.column_names_config.hid
        self.reweighting_config = reweighting_config
        self.control_variables_config = control_variables_config
        self.db = db
        self.reweighting_config = reweighting_config
        self.outer_iterations = (self.reweighting_config.outer_iterations)
        self.inner_iterations = (self.reweighting_config.inner_iterations)
        self.archive_performance_frequency = (
            self.reweighting_config.archive_performance_frequency)
        self.procedure = (
            self.reweighting_config.procedure)
        self.tolerance = (
            self.reweighting_config.tolerance)
        try:
            self.ds_format = (
                self.reweighting_config.ds_format)
        except ConfigError, e:
            print e
            self.ds_format = "full"
        # self.household_size_adjustment = household_size_adjustment

        self.sample_geo_names = self.column_names_config.sample_geo

        if self.sample_geo_names is None:
            pass
        elif isinstance(self.sample_geo_names, str):
            self.sample_geo_names = [self.sample_geo_names]
        else:
            self.sample_geo_names = \
                self.sample_geo_names.return_list()


        # self.sample_geo_names = sample_geo_names
        # print ("sample geo names is:{0}".format(self.sample_geo_names))
        # raw_input("quick check")
        self.adjust_household_ipf_config = adjust_household_ipf_config

    def create_ds_for_adjust_household(self):
        self.reweighting_ds_obj = Reweighting_DS(
            self.ds_format)

        geo_controls_config = self.control_variables_config.geo
        # print "GEO LEVEL DS"
        # (self.geo_stacked,
        # self.geo_row_idx,
        # self.geo_contrib) = (self._create_ds_for_adjust_household(
        #                      geo_controls_config))
        self.geo_stacked = self._create_ds_for_adjust_household(
            geo_controls_config)

        # self._create_sample_weights_df()
        self._create_reweighting_performance_df()

    def create_ds(self):
        self.reweighting_ds_obj = Reweighting_DS(
            self.ds_format)

        # print "PREPARING REGION LEVEL DS"
        region_controls_config = self.control_variables_config.region
        (self.region_stacked,
         self.region_row_idx,
         self.region_contrib) = (self._create_ds(region_controls_config, region_level=True))

        # print "GEO LEVEL DS"
        geo_controls_config = self.control_variables_config.geo
        (self.geo_stacked,
         self.geo_row_idx,
         self.geo_contrib) = (self._create_ds(geo_controls_config))

        self._create_reweighting_performance_df()

    def _create_ds_for_adjust_household(self, control_variables_config):
        housing_stacked_sample = \
            self._create_restruct_combined_across_housing_entities_adjust_household(
                control_variables_config)
        stacked_sample = housing_stacked_sample
        # row_idx, contrib = self.reweighting_ds_obj.get_row_idx(stacked_sample)
        # return (stacked_sample, row_idx, contrib)
        return stacked_sample

    def _create_restruct_combined_across_housing_entities_adjust_household(
            self, control_variables_config):
        # print "BEFORE COMBINING", self.sample_geo_names
        sample_restruct_list = []
        for entity in self.housing_entities:
            variable_names = (control_variables_config[entity]).return_list()
            sample = self.db.sample[entity]
            sample_restruct = self.reweighting_ds_obj.get_sample_restructure(
                entity, sample, variable_names, self.hid_name, self.sample_geo_names,
                self.adjust_household_ipf_config)
            sample_restruct_list.append(sample_restruct)

        stacked_sample = (self.reweighting_ds_obj.get_stacked_sample_restruct(
                          sample_restruct_list, self.sample_geo_names,
                          self.adjust_household_ipf_config))
        return stacked_sample

    def _create_ds(self, control_variables_config, region_level=False):

        housing_stacked_sample = \
            self._create_restruct_combined_across_entities(
                self.housing_entities, control_variables_config)

        if self.adjust_household_ipf_config is not None and region_level is False:
            housing_stacked_groupby_sample = self.reweighting_ds_obj.get_groupby(
                housing_stacked_sample, self.sample_geo_names,
                self.adjust_household_ipf_config, self.hid_name)
        else:
            housing_stacked_groupby_sample = housing_stacked_sample

        person_stacked_sample = \
            self._create_restruct_combined_across_entities(
                self.person_entities, control_variables_config)

        # print housing_stacked_groupby_sample.head()
        # print "_____________________________________"
        # print housing_stacked_groupby_sample.head()
        # print "_____________________________________"
        # print person_stacked_sample.head()

        stacked_sample = housing_stacked_groupby_sample.join(
            person_stacked_sample)

        row_idx, contrib = self.reweighting_ds_obj.get_row_idx(stacked_sample)
        return (stacked_sample, row_idx, contrib)

    def _create_restruct_combined_across_entities(self,
            entities, control_variables_config):
        # print "BEFORE COMBINING", self.sample_geo_names
        sample_restruct_list = []
        for entity in entities:
            if entity in self.housing_entities:
                sample_geo_names = self.sample_geo_names
                adjust_household_ipf_config = self.adjust_household_ipf_config
            else:
                sample_geo_names = None
                adjust_household_ipf_config = None
            variable_names = (control_variables_config[entity]).return_list()
            sample = self.db.sample[entity]
            sample_restruct = self.reweighting_ds_obj.get_sample_restructure(
                entity, sample, variable_names, self.hid_name, sample_geo_names,
                adjust_household_ipf_config)
            sample_restruct_list.append(sample_restruct)

        stacked_sample = (self.reweighting_ds_obj.get_stacked_sample_restruct(
                          sample_restruct_list, sample_geo_names,
                          adjust_household_ipf_config ))
        return stacked_sample

    """
    def _create_sample_weights_df(self):
        if self.adjust_household_ipf_config is not None:
            self.region_sample_weights = (pd.DataFrame(
                                          index=self.geo_stacked.index))
        else:
            self.region_sample_weights = (pd.DataFrame(
                                          index=self.region_stacked.index))
    """

    def _create_reweighting_performance_df(self):
        # TODO: In the future change the frequency at which
        # performance measures are stored as a parameter that is
        # specified by the user
        if self.archive_performance_frequency <= 0 :
            self.iters_to_archive = []
        else:
            self.iters_to_archive = range(0, self.outer_iterations,
                                          self.archive_performance_frequency)

        self.average_diffs = pd.DataFrame(index=self.db.geo_ids,
                                          columns=self.iters_to_archive)

    def _create_intermediate_sample_weights_df(
            self, region_id, filter_based_on_puma):
        geo_ids = self.db.get_geo_ids_for_region(region_id)
        len_geo_ids = len(geo_ids)

        sample_weights = np.ones((self.region_stacked.shape[0],
                                  len_geo_ids),
                                  dtype=float, order="C")

        # raw_input(
        #    "Filter based on puma value: {0}".format(filter_based_on_puma))

        if filter_based_on_puma == 0:
            # raw_input("filtering")
            return sample_weights

        geo_corr_to_sample = self.db.geo["geo_to_sample"]

        for index_geo_id, geo_id in enumerate(geo_ids):
            for index_sample_geo_name, sample_geo_name in enumerate(
                    self.sample_geo_names):
                sample_geo_ids = \
                    geo_corr_to_sample.loc[geo_id, sample_geo_name]

                if isinstance(sample_geo_ids, pd.Series):
                    sample_geo_ids = sample_geo_ids.values()
                else:
                    sample_geo_ids = [sample_geo_ids]

                filter_for_sample_geo_name = \
                    stacked[sample_geo_name].isin(sample_geo_ids)

                if index_sample_geo_name == 0:
                    filter_geo_id = filter_for_sample_geo_name
                else:
                    filter_geo_id = pd.logical_and(
                        filter_geo_id, filter_for_sample_geo_name)

            sample_weights[:, index_geo_id] = \
                filter_geo_id.values.astype(float)

        return sample_weights

    def _filter_stacked_sample_based_on_puma_for_geoid(self, stacked_sample, geo_id):
        geo_corr_to_sample = self.db.geo["geo_to_sample"]

        for index_sample_geo_name, sample_geo_name in enumerate(
                self.sample_geo_names):
            sample_geo_ids = \
                geo_corr_to_sample.loc[geo_id, sample_geo_name]

            if isinstance(sample_geo_ids, pd.Series):
                sample_geo_ids = sample_geo_ids.values()
            else:
                sample_geo_ids = [sample_geo_ids]

            filter_for_sample_geo_name = \
                stacked_sample[sample_geo_name].isin(sample_geo_ids)

            if index_sample_geo_name == 0:
                filter_geo_id = filter_for_sample_geo_name
            else:
                filter_geo_id = np.logical_and(
                    filter_geo_id, filter_for_sample_geo_name)
        return stacked_sample[filter_geo_id]
    """
    def _create_intermediate_sample_weights_df_adjust_household(
            self, region_id, filter_based_on_puma):
        geo_ids = self.db.get_geo_ids_for_region(region_id)
        len_geo_ids = len(geo_ids)

        geo_stacked_groupby = self.reweighting_ds_obj.get_groupby(
            self.geo_stacked, self.sample_geo_names, self.adjust_household_ipf_config)

        sample_weights_df = pd.DataFrame(index=geo_stacked_groupby.index)

        # sample_weights = np.ones((geo_stacked_groupby.shape[0],
        #                          len_geo_ids),
        #                          dtype=float, order="C")

        raw_input(
            "Filter based on puma value: {0}".format(filter_based_on_puma))

        if filter_based_on_puma == 0:
            raw_input("filtering")
            return sample_weights

        for index_geo_id, geo_id in enumerate(geo_ids):
            geo_stacked_filtered = \
                self._filter_stacked_sample_based_on_puma_for_geoid(
                    self.geo_stacked, geo_id)

            geo_stacked_groupby_filtered = self.reweighting_ds_obj.get_groupby(
                geo_stacked_filtered, self.sample_geo_names, self.adjust_household_ipf_config)

            sample_weights_df.loc[
                geo_stacked_groupby_filtered.index, geo_id] = 1
            sample_weights_df.fillna(0., inplace=True)

        print sample_weights_df
        print sample_weights_df.sum(axis=0)

        return sample_weights
    """

    def run_reweighting_adjust_household(self,
            geo_constraints, filter_based_on_puma=0):

        geo_stacked_groupby = self.reweighting_ds_obj.get_groupby(
            self.geo_stacked, self.sample_geo_names,
            self.adjust_household_ipf_config)
        sample_weights_df = pd.DataFrame(index=geo_stacked_groupby.index)

        self.region_sample_weights = \
            pd.DataFrame(index=geo_stacked_groupby.index)

        for region_id in self.db.region_ids:
            print ("\t%s for Region: %d" % (self.procedure, region_id))
            # print "\t\tConstraints sum:", geo_constraints.sum().sum()


            geo_ids = self.db.get_geo_ids_for_region(region_id)

            for index, geo_id in enumerate(geo_ids):
                # print ("\t\t\tGeo: %s " % geo_id)
                # print geo_constraints.loc[geo_id]
                if filter_based_on_puma == 0:
                    geo_stacked_filtered = geo_stacked_groupby
                    # raw_input("no filtering")
                else:
                    # raw_input("filtering")
                    geo_stacked_filtered = \
                        self._filter_stacked_sample_based_on_puma_for_geoid(
                            self.geo_stacked, geo_id)

                geo_stacked_groupby_filtered = self.reweighting_ds_obj.get_groupby(
                    geo_stacked_filtered, self.sample_geo_names,
                    self.adjust_household_ipf_config)
                geo_row_idx, geo_contrib = self.reweighting_ds_obj.get_row_idx(
                    geo_stacked_groupby_filtered)

                sample_weights = np.ones(
                    (geo_stacked_groupby_filtered.shape[0]))

                sample_weights_updated = (self._adjust_sample_weights(
                                          geo_id,
                                          sample_weights,
                                          geo_constraints.loc[geo_id],
                                          geo_row_idx,
                                          geo_contrib,
                                          iters=self.inner_iterations,
                                          geo=True))
                sample_weights_df.loc[
                    geo_stacked_groupby_filtered.index, geo_id] = \
                        sample_weights_updated
                """
                if iter in self.iters_to_archive:
                    self._calculate_populate_average_deviation(
                        geo_id, iter,
                        sample_weights,
                        geo_constraints.loc[geo_id], geo_row_idx)
                    pass
                """
            # print sample_weights_df
                # self._calculate_populate_average_deviation(
                #    geo_id, iter,
                #    sample_weights_updated,
                #    geo_constraints.loc[geo_id], geo_contrib, geo_row_idx)

                # raw_input("Check results")
            self._populate_sample_weights(sample_weights_df.values, region_id, geo_ids)
        print "\t\tSample_weights sum after household adjustment:", self.region_sample_weights.sum().sum()
        # raise Exception

    def run_reweighting(self, region_constraints,
                        geo_constraints, filter_based_on_puma=0):

        self.region_sample_weights = \
            pd.DataFrame(index=self.region_stacked.index)

        for region_id in self.db.region_ids:
            print ("\t%s for Region: %d" % (self.procedure, region_id))

            sample_weights = self._create_intermediate_sample_weights_df(
                region_id, filter_based_on_puma)

            geo_ids = self.db.get_geo_ids_for_region(region_id)

            # print "Outer iterations", self.outer_iterations
            for iter in range(self.outer_iterations):
                t = time.time()
                print "\t\tRegion: %s and Iter: %s" % (region_id, iter)
                if region_constraints is not None:
                    sample_weights = (self._adjust_sample_weights(
                                      region_id, sample_weights,
                                      region_constraints.loc[region_id],
                                      self.region_row_idx,
                                      self.region_contrib))
                # print "After region:", sample_weights[:, :4]
                # raw_input("region_done")
                for index, geo_id in enumerate(geo_ids):
                    # print ("\t\t\tGeo: %s " % geo_id)
                    # print geo_constraints.loc[geo_id]
                    # t_i = time.time()
                    sample_weights_updated = (self._adjust_sample_weights(
                                                geo_id,
                                                sample_weights[:, index],
                                                geo_constraints.loc[geo_id],
                                                self.geo_row_idx,
                                                self.geo_contrib,
                                                iters=self.inner_iterations,
                                                geo=True))
                    # t_u = time.time()
                    sample_weights[:, index] = sample_weights_updated
                    # print ("""\t\t\t\tUpdated weights after one round of """
                    #        """inner iterations in %.4f""" %(time.time() - t_u))
                    # print "After geo:", sample_weights[:, :4]
                    # print ("sample_weights sum: %.6f" % (
                    #    sample_weights[:, index].sum()))
                    if iter in self.iters_to_archive:
                        self._calculate_populate_average_deviation(
                            geo_id, iter,
                            sample_weights_updated,
                            geo_constraints.loc[geo_id], self.geo_contrib, self.geo_row_idx)
                        pass
                    # raw_input("Geo done %s" %geo_id)
                    # print ("""\t\t\tOne round of inner iterations for """
                    #        """geo: %s complete in %.4f""" %
                    #        (geo_id, time.time() - t_i))

                # print ("\t\t\tOne outer iteration complete in %.4f" %
                #       (time.time() - t))
            self._populate_sample_weights(sample_weights, region_id, geo_ids)
            # print self.average_deviations
            print "\t\tSample_weights sum:", sample_weights.sum()

    def _adjust_sample_weights(self, geo_id, sample_weights, constraints,
                               row_idx, contrib, iters=1, geo=False):
        # print "\t\t\tAdjustemnt with respect to geo flag is:{0}".format(geo)

        constraints_filtered = constraints
        if self.procedure == "ipu":
            return self._ipu_adjust_sample_weights(
                geo_id, sample_weights, constraints_filtered,
                row_idx, contrib, iters, geo)
        elif self.procedure == "entropy":
            return self._entropy_adjust_sample_weights(
                geo_id, sample_weights, constraints_filtered,
                row_idx, contrib, iters, geo)
        elif self.procedure == "lsq":
            return self._lsq_adjust_sample_weights(
                geo_id, sample_weights, constraints_filtered,
                row_idx, contrib, iters, geo)

    def _ipu_adjust_sample_weights(self, geo_id, sample_weights, constraints,
                                   row_idx, contrib,
                                   iters=1, geo=False):
        """
        if geo:
            row_idx = self.geo_row_idx
            contrib = self.geo_contrib
        else:
            row_idx = self.region_row_idx
            contrib = self.region_contrib
        """
        # t = time.time()
        sample_weights = np.array(sample_weights, order="C")
        # t_create_array = time.time() - t
        # print "\t\t\t\tCreating array for one round of inner iters takes %.4f" %(t_create_array)

        # print "Number if iters: {0}".format(iters)

        for i in range(iters):
            # print ("This is iter:{0}".format(i))
            for column in reversed(constraints.index):
                if constraints[column] == 0:
                    # print ("\t\t\t\tSkipping column: {0} with constraint {1}".format(
                    #       column, constraints[column]))
                    pass
                # TODO: the reversed iteration of list needs to be replaced
                # with a user specified ordering of the constraints

                # if column == 'person_dummy1':
                #    print "before adjusting wrt person_dummy"
                #    self._print_calculate_average_deviation(
                #            geo_id, iter, sample_weights, constraints, contrib, row_idx)
                #    raw_input("\t\t\t\t\tFor column: {0} with constraint {1} enter".format(
                #        column, constraints[column]))

                # print "\t\t\t\t", column

                if self.ds_format == "full":
                    # t = time.time()
                    if geo is False:
                        weighted_sum = (
                            sample_weights.T.dot(contrib[column])
                            ).sum()
                    else:
                        weighted_sum = sample_weights.dot(contrib[column])
                    # t_weighted_sum += (time.time() - t)

                if self.ds_format == "contributing":
                    # t = time.time()
                    sample_weights_for_column = sample_weights[row_idx[column]]
                    if geo is False:
                        weighted_sum2 = (
                            sample_weights_for_column.T.dot(contrib[column])
                            ).sum()
                    else:
                        weighted_sum2 = sample_weights_for_column.dot(
                            contrib[column])
                    # t_weighted_sum2 += (time.time() - t)
                    weighted_sum = weighted_sum2

                """
                if abs(weighted_sum - weighted_sum2) < self.tolerance:
                    pass
                else:
                    print weighted_sum, weighted_sum2
                """
                if weighted_sum == 0:
                    # print ("""\t\t\t\tWeighted sum for column {0} with """
                    #       """constraint {1} in iter {2} """
                    #       """is zero so don't adjust""".format(
                    #            column, constraints[column], i))
                    continue

                adjustment = constraints[column]/weighted_sum
                # t = time.time()
                sample_weights[row_idx[column]] *= adjustment
                # t_slicing += (time.time() - t)
        # print "\t\t\t\tWeighted sum for one round of inner iters takes %.4f" %(t_weighted_sum)
        # print "\t\t\t\tWeighted sum 2 for one round of inner iters takes %.4f" %(t_weighted_sum2)
        # print "\t\t\t\tSlicing for one round of inner iters takes %.4f" %(t_slicing)
                if np.isnan(sample_weights).any():
                    print sample_weights
                    print ("Adjusting wrt {0} and constraint is: {1} and adjustment is: {2}".format(
                        column, constraints[column], adjustment))
                    raw_input()

                # if column == 'person_dummy':
                #    print "AFTER adjusting wrt person_dummy"
                #    self._print_calculate_average_deviation(
                #            geo_id, iter, sample_weights, constraints, contrib, row_idx)
                #    raw_input("\t\t\t\t\tFor column: {0} with constraint {1} enter".format(
                #        column, constraints[column]))

            # self._print_calculate_average_deviation(
            #        geo_id, i, sample_weights, constraints, contrib, row_idx)
            # print "\t\tAfter iter: {0} sample_weights sum is: {1}, constrain sum is: {2}".format(i, sample_weights.sum(), constraints.sum())
            # raw_input("\tafter iter: {0}".format(i))
        # if not geo is False:
        #    self._print_calculate_average_deviation(
        #        geo_id, i, sample_weights, constraints, contrib, row_idx)

        # print "Shape of the solution is: {0}".format(sample_weights.shape)
        # raw_input()
        # print "last constraint is: {0}".format(column)
        # print constraints
        # print "\t\t\t\tConstraint sum is: {0}".format(constraints.sum())
        # print "\t\t\t\tWeighted sum is: {0}".format(sample_weights.sum())
        # raw_input()
        return sample_weights

    def _lsq_adjust_sample_weights(self, geo_id, sample_weights, constraints,
                                   row_idx, contrib,
                                   iters=1, geo=False):
        # t = time.time()
        # sample_weights = np.array(sample_weights, order="C")
        # t_create_array = time.time() - t
        # print "\t\t\t\tCreating array for one round of inner iters takes %.4f" %(t_create_array)
        # constraint = contrib * W


        contrib_list = []
        for column in (constraints.index):
            contrib_list.append(contrib[column])
        contrib_matrix = np.array(contrib_list)
        contrib_df = pd.DataFrame(contrib_matrix, columns=constraints.index.tolist())
        contrib_df.to_csv("contrib_matrix.csv")
        constraints_array = constraints.values

        # sample_weights = np.linalg.lstsq(contrib_matrix, constraints_array)
        # print sample_weights

        constraints.to_csv("{0}_constraints.csv".format(geo_id))

        from scipy.optimize import lsq_linear
        sample_weights2 = lsq_linear(
            contrib_matrix, constraints_array, bounds = (0, np.inf),
            method = "bvls", lsq_solver = "exact",
            lsmr_tol = None, verbose=0, max_iter=200)
        # print sample_weights2

        # print "Shape of the solution is: {0}".format(sample_weights2.x.shape)
        # i=0
        # self._calculate_populate_average_deviation(
        #            geo_id, i, sample_weights2.x, constraints, contrib, row_idx)

        # print constraints_array
        # print (constraints.index)
        # print "Average deviation is: {0}".format(np.mean(np.absolute(sample_weights2.fun)))
        # print "\t\tAfter iter: {0} sample_weights sum is: {1}, constrain sum is: {2}".format(i, sample_weights.sum(), constraints.sum())
        # raw_input("\tafter lsq is complete")
        # print "Shape of the solution is: {0}".format(sample_weights2.x.shape)
        # raw_input()

        return sample_weights2.x

    def _entropy_adjust_sample_weights(self, geo_id,
                                       sample_weights, constraints,
                                       row_idx, contrib,
                                       iters=1, geo=False):

        if geo:
            # row_idx = self.geo_row_idx
            # contrib = self.geo_contrib
            pass
        else:
            # row_idx = self.region_row_idx
            # contrib = self.region_contrib
            ones_array = np.ones((sample_weights.shape[1]), order="C")

        # t = time.time()
        sample_weights = np.array(sample_weights, order="C")
        for i in range(iters):
            for column in (constraints.index):
                if constraints[column] == 0:
                    # print ("\t\t\t\tSkipping column: {0} with constraint {1}".format(
                    #       column, constraints[column]))
                    continue

                # TODO: the reversed iteration of list needs to be replaced
                # with a user specified ordering of the constraints
                if geo is False:
                    weights_mul_contrib = (
                        np.dot(sample_weights, ones_array) * contrib[column])
                else:
                    weights_mul_contrib = sample_weights * contrib[column]

                root = self._find_root(
                    contrib[column], constraints[column], weights_mul_contrib)
                # print "This is root:{0}".format(root)
                if root == -1:
                    # print ("""\t\t\t\tWeighted sum for column {0} with """
                    #       """constraint {1} in iter {2} """
                    #       """is zero so don't adjust""".format(
                    #            column, constraints[column], i))
                    continue

                adjustment = root**contrib[column]
                sample_weights[row_idx[column]] = np.multiply(
                    sample_weights[row_idx[column]].T,
                    adjustment[row_idx[column]]).T
                # print "column:{0}, adjustment:{1}".format(column, root)
                # if column == ('person_dummy', 1):
                #    print "\t\t\tafter adjusting wrt person_dummy"
                #    self._calculate_populate_average_deviation(
                #            geo_id, i, sample_weights, constraints)
                    # print "adjustment:{0}".format(root)
                    # raw_input()

        return sample_weights

    def _find_equation(self, contrib, weights_mul_contrib):
        root_power_weight = np.bincount(contrib, weights=weights_mul_contrib)
        root_power = np.array(range(contrib.max() + 1))
        return root_power[1:], root_power_weight[1:]

    def _optimizing_function(self, root, root_power, root_power_weight,
                             constraint):
        function_value = (
            root_power_weight.dot(root ** root_power) - constraint)
        return function_value

    def _find_root(self, contrib, constraint, weights_mul_contrib):
        root_power, root_power_weight = self._find_equation(
            contrib, weights_mul_contrib)
        if root_power_weight.size == 1:
            if root_power_weight == 0:
                return -1

        if len(root_power) == 1:
            root = constraint/root_power_weight
        else:
            starting_value = 0.0
            root = fsolve(
                self._optimizing_function, starting_value, args=(
                    root_power, root_power_weight, constraint))
        return root

    def _calculate_populate_average_deviation(
                self, geo_id, iter, sample_weights, constraints, contrib, row_idx):
        diff_sum = 0
        sample_weights = np.array(sample_weights, order="C")
        for column in constraints.index:
            if constraints[column] == 0:
                continue

            if self.ds_format == "full":
                weighted_sum = sample_weights.dot(contrib[column])
            elif self.ds_format == "contributing":
                sample_weights_for_column = sample_weights[row_idx[column]]
                weighted_sum = sample_weights_for_column.dot(
                    contrib[column])

            diff = weighted_sum - constraints[column]
            diff_sum += np.abs(diff)
        average_diff = diff_sum/constraints.shape[0]
        self.average_diffs.loc[geo_id, iter] = average_diff

    def _print_calculate_average_deviation(
                self, geo_id, iter, sample_weights, constraints, contrib, row_idx):
        diff_sum = 0
        sample_weights = np.array(sample_weights, order="C")
        for column in constraints.index:
            if constraints[column] == 0:
                continue

            if self.ds_format == "full":
                weighted_sum = sample_weights.dot(contrib[column])
            elif self.ds_format == "contributing":
                sample_weights_for_column = sample_weights[row_idx[column]]
                weighted_sum = sample_weights_for_column.dot(
                    contrib[column])

            diff = weighted_sum - constraints[column]
            diff_sum += np.abs(diff)
            print ("""\t\t\t\tcolumn:{0}, constraint:{1}, """
                   """weighted_sum:{2}, diff:{3}""".format(
                        column, constraints[column], weighted_sum, diff))
        average_diff = diff_sum/constraints.shape[0]
        print "\t\t\tAverage absolute diff:{0} in iter:{1}".format(
            average_diff, iter)

    def _populate_sample_weights(self, sample_weights, region_id, geo_ids):
        for index, geo_id in enumerate(geo_ids):
            # self.region_sample_weights[(region_id,
            #                            geo_id)] = sample_weights[:, index]
            self.region_sample_weights[geo_id] = sample_weights[:, index]

    def _transform_column_index(self):
        multi_index = (pd.MultiIndex.from_tuples(
                       self.region_sample_weights.columns.values,
                       names=["region_id", "geo_id"]))
        self.region_sample_weights.columns = multi_index
