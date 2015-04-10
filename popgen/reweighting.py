import numpy as np
import pandas as pd


#TODO: Reimplement all DS processing in the Syn_Population Class
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
            contrib[column] = sample_restruct[column].values
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
        return stacked_sample


class Run_IPU(object):
    def __init__(self, entities, column_names_config, scenario_config, db):
        self.entities = entities
        self.column_names_config = column_names_config
        self.scenario_config = scenario_config
        self.db = db
        self.outer_iterations = (self.scenario_config
                                 .parameters.reweighting.outer_iterations)
        self.inner_iterations = (self.scenario_config
                                 .parameters.reweighting.inner_iterations)

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

    def _create_ds_for_resolution(self, control_variables_config):
        sample_restruct_list = []
        ipu_ds_obj = Reweighting_DS()

        hid_name = self.column_names_config.hid

        for entity in self.entities:
            variable_names = (control_variables_config[entity]).return_list()
            sample = self.db.sample[entity]
            sample_restruct = ipu_ds_obj.get_sample_restructure(entity,
                                                                sample,
                                                                variable_names,
                                                                hid_name)
            sample_restruct_list.append(sample_restruct)

        stacked_sample = (ipu_ds_obj.get_stacked_sample_restruct(
                          sample_restruct_list))
        row_idx, contrib = ipu_ds_obj.get_row_idx(stacked_sample)
        #print "Sample stacked\n", stacked_sample[:10]
        return (stacked_sample, row_idx, contrib)

    def _create_sample_weights_df(self):
        self.region_sample_weights = (pd.DataFrame(
                                      index=self.region_stacked.index))

    def run_ipu(self, region_constraints, geo_constraints):
        for region_id in self.db.region_ids:
            print ("IPU for Region: %d" % region_id)
            geo_ids = self.db.get_geo_ids_for_region(region_id)
            len_geo_ids = len(geo_ids)
            sample_weights = np.ones((self.region_stacked.shape[0],
                                      len_geo_ids),
                                     dtype=float, order="C")
            #print "Outer iterations", self.outer_iterations
            for iter in range(self.outer_iterations):
                #print "Region: %s and Iter: %s" % (region_id, iter)
                if region_constraints is not None:
                    sample_weights = (self._adjust_sample_weights(
                                      sample_weights,
                                      region_constraints.loc[region_id]))
                #print "After region:", sample_weights[:, :4]

                for index, geo_id in enumerate(geo_ids):
                    #print ("Geo: %s " % geo_id)
                    sample_weights[:, index] = (self._adjust_sample_weights(
                                                sample_weights[:, index],
                                                geo_constraints.loc[geo_id],
                                                iters=self.inner_iterations,
                                                geo=True))
                #print "After geo:", sample_weights[:, :4]

            self._populate_sample_weights(sample_weights, region_id, geo_ids)
            print "sample_weights sum:", sample_weights.sum()

    def _adjust_sample_weights(self, sample_weights, constraints,
                               iters=1, geo=False):
        if geo:
            row_idx = self.geo_row_idx
            contrib = self.geo_contrib
        else:
            row_idx = self.region_row_idx
            contrib = self.region_contrib

        sample_weights = np.ascontiguousarray(sample_weights)

        for i in range(iters):
            for column in reversed(constraints.index):
                #TODO: the reversed iteration of list needs to be replaced with
                #a user specified ordering of the constraints
                if geo is False:
                    weighted_sum = (sample_weights
                                    .sum(axis=1).dot(contrib[column]))
                else:
                    weighted_sum = sample_weights.dot(contrib[column])
                adjustment = constraints[column]/weighted_sum
                sample_weights[row_idx[column]] *= adjustment
        return sample_weights

    def _populate_sample_weights(self, sample_weights, region_id, geo_ids):
        for index, geo_id in enumerate(geo_ids):
            #self.region_sample_weights[(region_id,
            #                            geo_id)] = sample_weights[:, index]
            self.region_sample_weights[geo_id] = sample_weights[:, index]

    def _transform_column_index(self):
        multi_index = (pd.MultiIndex.from_tuples(
                       self.region_sample_weights.columns.values,
                       names=["region_id", "geo_id"]))
        self.region_sample_weights.columns = multi_index
