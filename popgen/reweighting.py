import numpy as np
import pandas as pd

import time

from scipy.optimize import fsolve, newton

# from sympy.abc import x as root
# from sympy import solve


# TODO: Move all DS processing to Syn_Population Class
class Reweighting_DS(object):
    def __init__(self):
        pass

    def get_sample_restructure(self, entity, sample, variable_names, hid_name):
        sample["entity"] = entity
        groupby_columns = [hid_name, "entity"] + variable_names
        columns_count = len(groupby_columns)
        sample_restruct = (sample.groupby(groupby_columns)
                           .size()
                           .unstack(level=range(1, columns_count))
                           .fillna(0)
                           )
        return sample_restruct

    def get_row_idx(self, sample_restruct):
        row_idx = {}
        contrib = {}
        for column in sample_restruct.columns.values.tolist():
            rows = np.where(sample_restruct[column] > 0)[0]
            row_idx[column] = rows
            contrib[column] = np.array(
                sample_restruct[column].values, order="C", dtype=int)
        return (row_idx, contrib)

    def get_stacked_sample_restruct(self, sample_restruct_list):
        if len(sample_restruct_list) == 0:
            return None
        elif len(sample_restruct_list) == 1:
            return sample_restruct_list[0]

        stacked_sample = sample_restruct_list[0]
        for i, sample_restruct in enumerate(sample_restruct_list[1:]):
            len_left_frame_index = len(stacked_sample.columns.values[0])
            len_right_frame_index = len(sample_restruct.columns.values[0])

            if len_left_frame_index >= len_right_frame_index:
                stacked_sample = stacked_sample.join(sample_restruct,
                                                     how="outer").fillna(0)
            else:
                stacked_sample = sample_restruct.join(stacked_sample,
                                                      how="outer").fillna(0)
        stacked_sample.sort(inplace=True)  # Sort by row indices
        stacked_sample.sort_index(axis=1,
                                  inplace=True)  # Sort columns alphabetically
        stacked_sample.columns = pd.Index(stacked_sample.columns,
                                          tuplelize_cols=False)
        return stacked_sample


class Run_Reweighting(object):
    def __init__(self, entities, column_names_config, scenario_config, db):
        self.entities = entities
        self.column_names_config = column_names_config
        self.scenario_config = scenario_config
        self.db = db
        self.outer_iterations = (self.scenario_config
                                 .parameters.reweighting.outer_iterations)
        self.inner_iterations = (self.scenario_config
                                 .parameters.reweighting.inner_iterations)
        self.archive_performance_frequency = (
            self.scenario_config.parameters
            .reweighting.archive_performance_frequency)
        self.procedure = (
            self.scenario_config.parameters.reweighting.procedure)

    def create_ds(self):
        region_controls_config = self.scenario_config.control_variables.region
        (self.region_stacked,
         self.region_row_idx,
         self.region_contrib) = (self._create_ds_for_resolution(
                                 region_controls_config))
        geo_controls_config = self.scenario_config.control_variables.geo
        (self.geo_stacked,
         self.geo_row_idx,
         self.geo_contrib) = (self._create_ds_for_resolution(
                              geo_controls_config))
        self._create_sample_weights_df()
        self._create_reweighting_performance_df()

    def _create_ds_for_resolution(self, control_variables_config):
        sample_restruct_list = []
        reweighting_ds_obj = Reweighting_DS()

        hid_name = self.column_names_config.hid

        for entity in self.entities:
            variable_names = (control_variables_config[entity]).return_list()
            sample = self.db.sample[entity]
            sample_restruct = reweighting_ds_obj.get_sample_restructure(
                entity, sample, variable_names, hid_name)
            sample_restruct_list.append(sample_restruct)

        stacked_sample = (reweighting_ds_obj.get_stacked_sample_restruct(
                          sample_restruct_list))
        row_idx, contrib = reweighting_ds_obj.get_row_idx(stacked_sample)
        # print "Sample stacked\n", stacked_sample[:10]
        return (stacked_sample, row_idx, contrib)

    def _create_sample_weights_df(self):
        self.region_sample_weights = (pd.DataFrame(
                                      index=self.region_stacked.index))

    def _create_reweighting_performance_df(self):
        # TODO: In the future change the frequency at which
        # performance measures are stored as a parameter that is
        # specified by the user
        self.iters_to_archive = range(0, self.outer_iterations,
                                      self.archive_performance_frequency)
        self.average_diffs = pd.DataFrame(index=self.db.geo_ids,
                                          columns=self.iters_to_archive)

    def run_reweighting(self, region_constraints, geo_constraints):
        for region_id in self.db.region_ids:
            print ("\t%s for Region: %d" % (self.procedure, region_id))
            geo_ids = self.db.get_geo_ids_for_region(region_id)
            len_geo_ids = len(geo_ids)
            sample_weights = np.ones((self.region_stacked.shape[0],
                                      len_geo_ids),
                                     dtype=float, order="C")
            # print "Outer iterations", self.outer_iterations
            for iter in range(self.outer_iterations):
                t = time.time()
                # print "Region: %s and Iter: %s" % (region_id, iter)
                if region_constraints is not None:
                    sample_weights = (self._adjust_sample_weights(
                                      sample_weights,
                                      region_constraints.loc[region_id]))
                # print "After region:", sample_weights[:, :4]
                # raw_input("region_done")
                for index, geo_id in enumerate(geo_ids):
                    # print ("Geo: %s " % geo_id)
                    sample_weights[:, index] = (self._adjust_sample_weights(
                                                sample_weights[:, index],
                                                geo_constraints.loc[geo_id],
                                                iters=self.inner_iterations,
                                                geo=True))
                    # print "After geo:", sample_weights[:, :4]
                    # print ("sample_weights sum: %.6f" % (
                    #    sample_weights[:, index].sum()))
                    if iter in self.iters_to_archive:
                        self._calculate_populate_average_deviation(
                            geo_id, iter,
                            sample_weights[:, index],
                            geo_constraints.loc[geo_id])
                        pass
                    # raw_input("Geo done %s" %geo_id)
                # print ("\t\t\tOne outer iteration complete in %.4f" %
                #       (time.time() - t))
            self._populate_sample_weights(sample_weights, region_id, geo_ids)
            # print self.average_deviations
            print "\tsample_weights sum:", sample_weights.sum()

    def _adjust_sample_weights(self, sample_weights, constraints,
                               iters=1, geo=False):
        if self.procedure == "ipu":
            return self._ipu_adjust_sample_weights(
                sample_weights, constraints, iters, geo)
        elif self.procedure == "entropy":
            return self._entropy_adjust_sample_weights(
                sample_weights, constraints, iters, geo)

    def _ipu_adjust_sample_weights(self, sample_weights, constraints,
                                   iters=1, geo=False):
        if geo:
            row_idx = self.geo_row_idx
            contrib = self.geo_contrib
        else:
            row_idx = self.region_row_idx
            contrib = self.region_contrib
        # t = time.time()
        sample_weights = np.array(sample_weights, order="C")
        for i in range(iters):
            for column in reversed(constraints.index):
                # TODO: the reversed iteration of list needs to be replaced
                # with a user specified ordering of the constraints
                if geo is False:
                    weighted_sum = (
                        sample_weights.T.dot(contrib[column])
                        ).sum()
                else:
                    weighted_sum = sample_weights.dot(contrib[column])

                if weighted_sum == 0:
                    print ("""Weighted sum for column %s in iter %d"""
                           """is zero so don't adjust""" % (column, i))
                    continue

                adjustment = constraints[column]/weighted_sum
                sample_weights[row_idx[column]] *= adjustment

        return sample_weights

    def _entropy_adjust_sample_weights(self, sample_weights, constraints,
                                       iters=1, geo=False):
        if geo:
            row_idx = self.geo_row_idx
            contrib = self.geo_contrib
        else:
            row_idx = self.region_row_idx
            contrib = self.region_contrib
            ones_array = np.ones((sample_weights.shape[1]), order="C")

        # t = time.time()
        sample_weights = np.array(sample_weights, order="C")
        for i in range(iters):
            for column in reversed(constraints.index):
                # TODO: the reversed iteration of list needs to be replaced
                # with a user specified ordering of the constraints
                if geo is False:
                    weights_mul_contrib = (
                        np.dot(sample_weights, ones_array) * contrib[column])
                else:
                    weights_mul_contrib = sample_weights * contrib[column]

                root = self._find_root(
                    contrib[column], constraints[column], weights_mul_contrib)
                adjustment = root**contrib[column]
                sample_weights[row_idx[column]] = np.multiply(
                    sample_weights[row_idx[column]].T,
                    adjustment[row_idx[column]]).T

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

        if len(root_power) == 1:
            root = constraint/root_power_weight
        else:
            starting_value = 0.0
            root = fsolve(
                self._optimizing_function, starting_value, args=(
                    root_power, root_power_weight, constraint))
        return root

    def _calculate_populate_average_deviation(
            self, geo_id, iter, sample_weights, constraints):
        diff_sum = 0

        sample_weights = np.array(sample_weights, order="C")
        for column in constraints.index:
            weighted_sum = sample_weights.dot(self.geo_contrib[column])
            diff_sum += (np.abs(weighted_sum - constraints[column]) /
                         constraints[column])
        average_diff = diff_sum/constraints.shape[0]
        self.average_diffs.loc[geo_id, iter] = average_diff

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
