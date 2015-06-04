import copy

import numpy as np
import pandas as pd


# TODO: Move all DS processing in the Syn_Population Class
class IPF_DS(object):
    def __init__(self, sample, entity, variable_names,
                 variables_count, variables_cats,
                 sample_geo_name=None):
        self.sample = sample
        self.entity = entity
        self.variable_names = variable_names
        self.variables_count = variables_count
        self.variables_cats = variables_cats
        self.sample_geo_name = sample_geo_name

    def get_seed(self):
        groupby_columns = ["entity"] + self.variable_names
        if self.sample_geo_name is not None:
            groupby_columns = ([self.sample_geo_name] +
                               groupby_columns)
        self.sample["entity"] = self.entity
        seed = (self.sample.groupby(groupby_columns)
                .size().astype(float))
        seed.name = "frequency"
        seed = seed.reset_index()
        seed.set_index(keys=groupby_columns,
                       drop=False, inplace=True)
        return seed

    def get_row_idx(self, seed):
        row_idx = {}
        for index, var in enumerate(self.variable_names):
            for cat in self.variables_cats[var]:
                row_idx[(var, cat)] = seed[var].values == cat
        return row_idx


class IPF(object):
    def __init__(self, seed_all, seed, idx, marginals, ipf_config,
                 variable_names, variables_cats, variables_cats_count,
                 ):
        self.seed_all = seed_all
        self.seed = seed
        self.idx = idx
        self.marginals = marginals
        self.ipf_config = ipf_config
        self.variable_names = variable_names
        self.variables_cats = variables_cats
        self.variables_cats_count = variables_cats_count
        self.ipf_iters = self.ipf_config.iterations
        self.ipf_tolerance = self.ipf_config.tolerance
        self.zero_marginal_correction = (
            self.ipf_config.zero_marginal_correction)
        self.archive_performance_frequency = (
            self.ipf_config.archive_performance_frequency)
        self.average_diff_iters = []
        self.iter_convergence = None

    def run_ipf(self):
        self.frequencies = self._correct_zero_cell_issue()
        # self.frequencies = self.seed["frequency"].values
        for c_iter in xrange(self.ipf_iters):
            # print "Iter:", c_iter
            self._adjust_cell_frequencies()
            # Checks for convergence every 5 iterations

            # TODO: In the future change the frequency at which
            # performance measures are stored as a parameter that is
            # specified by the user

            if (c_iter % self.archive_performance_frequency) == 0:
                if self._check_convergence():
                    # print "\t\t\tConvergence achieved in %d iter" % (c_iter)
                    self.iter_convergence = c_iter
                    break

    def _correct_zero_cell_issue(self):
        if self.seed.shape[0] != self.seed_all.shape[0]:
            self.seed_all["prob"] = (self.seed["frequency"] /
                                     self.seed["frequency"].sum())
            null_rows = self.seed_all["prob"].isnull()
            self.seed_all["prob_all"] = (self.seed_all["frequency"] /
                                         self.seed_all["frequency"].sum())
            self.seed_all.loc[null_rows, "prob"] = (
                self.seed_all.loc[null_rows, "prob_all"])
            borrowed_sum = self.seed_all.loc[null_rows, "prob"].sum()
            adjustment = 1 - borrowed_sum
            self.seed_all.loc[~null_rows, "prob"] *= adjustment

            return self.seed_all["prob"].copy().values
        else:
            return self.seed["frequency"].copy().values

    def _adjust_cell_frequencies(self):
        for var in self.variable_names:
            for cat in self.variables_cats[var]:
                row_subset = self.idx[(var, cat)]
                # TODO: There is a possible issue with how Pandas reads csv
                # files with headers. In the notebook implementation the
                # multiindex is read as alphanumeric whereas in this
                # implementation it is being read as a "alpha" only
                # the below indexing for marginals is just a hack ... need
                # to replace
                marginal = self.marginals.loc[(var, "%s" % cat)]
                if marginal == 0:
                    marginal = self.zero_marginal_correction

                adjustment = (marginal /
                              self.frequencies[row_subset].sum())
                self.frequencies[row_subset] *= adjustment

                if (self.frequencies == 0).any():
                    cells_zero_values = self.frequencies == 0
                    self.frequencies[cells_zero_values] = (
                        np.finfo(np.float64).tiny)

    def _check_convergence(self):
        average_diff = self._calculate_average_deviation()
        self.average_diff_iters.append(average_diff)
        if len(self.average_diff_iters) > 1:
            if (np.abs(self.average_diff_iters[-1] -
                       self.average_diff_iters[-2]) < self.ipf_tolerance):
                return True
        return False

    def _print_marginals(self):
        for var in self.variable_names:
            for cat in self.variables_cats[var]:
                row_subset = self.idx[(var, cat)]
                adjusted_frequency = self.frequencies[row_subset].sum()
                original_frequency = self.marginals.loc[(var, "%s" % cat)]
                print ("\t", (var, "%s" % cat),
                       original_frequency, adjusted_frequency)

    def _calculate_average_deviation(self):
        diff_sum = 0
        for var in self.variable_names[:-1]:
            for cat in self.variables_cats[var]:
                row_subset = self.idx[(var, cat)]
                adjusted_frequency = self.frequencies[row_subset].sum()
                # TODO: See above to-do same fix here
                original_frequency = self.marginals.loc[(var, "%s" % cat)]
                if original_frequency == 0:
                    original_frequency = self.zero_marginal_correction
            diff_sum += (np.abs(adjusted_frequency - original_frequency) /
                         original_frequency)
        average_diff = diff_sum/self.variables_cats_count
        # print "Average Deviation", average_diff
        return average_diff


class Run_IPF(object):
    def __init__(self, entities, housing_entities,
                 column_names_config, scenario_config, db):
        self.entities = entities
        self.housing_entities = housing_entities
        self.column_names_config = column_names_config
        self.scenario_config = scenario_config
        self.db = db
        self.ipf_config = self.scenario_config.parameters.ipf
        self.ipf_rounding = self.ipf_config.rounding_procedure
        self.sample_geo_name = self.column_names_config.sample_geo

    def run_ipf(self):
        region_marginals = self.db.region_marginals
        region_controls_config = self.scenario_config.control_variables.region
        region_ids = self.db.region_ids
        region_to_sample = self.db.geo["region_to_sample"]
        (self.region_constraints,
         self.region_constraints_dict,
         self.region_iters_convergence_dict,
         self.region_average_diffs_dict) = (self._run_ipf_for_resolution(
                                            region_marginals,
                                            region_controls_config,
                                            region_ids, region_to_sample))
        self.region_columns_dict = (self._get_columns_constraints_dict(
                                    self.region_constraints_dict))

        geo_marginals = self.db.geo_marginals
        geo_controls_config = self.scenario_config.control_variables.geo
        geo_ids = self.db.geo_ids
        geo_to_sample = self.db.geo["geo_to_sample"]
        (self.geo_constraints,
         self.geo_constraints_dict,
         self.geo_iters_convergence_dict,
         self.geo_average_diffs_dict) = (self._run_ipf_for_resolution(
                                         geo_marginals,
                                         geo_controls_config,
                                         geo_ids, geo_to_sample))
        self.geo_columns_dict = (self._get_columns_constraints_dict(
                                 self.geo_constraints_dict))

        if self.ipf_rounding == "bucket":
            self.geo_frequencies = (self._get_frequencies_for_resolution(
                                    geo_ids, self.geo_constraints_dict,
                                    "bucket"))

    def _run_ipf_for_resolution(self, marginals_at_resolution,
                                control_variables_config,
                                geo_ids, geo_corr_to_sample):
        constraints_list = []
        constraints_dict = {}
        iters_convergence_dict = {}
        average_diffs_dict = {}
        for entity in self.entities:
            print ("\tIPF for Entity: %s complete" % entity)

            sample = self.db.sample[entity]
            marginals = marginals_at_resolution[entity]
            variable_names = control_variables_config[entity].return_list()

            if len(variable_names) == 0:
                continue

            variables_cats = (self.db.return_variables_cats(entity,
                                                            variable_names))
            variables_count = len(variable_names)
            variables_cats_count = sum([len(cats) for cats in
                                        variables_cats.values()])
            (seed_geo,
             seed_all,
             row_idx) = (self._create_ds_for_resolution_entity(
                         sample, entity, variable_names,
                         variables_count, variables_cats,
                         self.sample_geo_name))

            (constraints,
             iters_convergence,
             average_diffs) = self._run_ipf_all_geos(entity, seed_geo,
                                                     seed_all,
                                                     row_idx, marginals,
                                                     variable_names,
                                                     variables_count,
                                                     variables_cats,
                                                     variables_cats_count,
                                                     geo_ids,
                                                     geo_corr_to_sample)
            constraints_dict[entity] = constraints
            iters_convergence[entity] = iters_convergence
            average_diffs_dict[entity] = average_diffs
            constraints_list.append(constraints)
        constraints_resolution = (self._get_stacked_constraints(
                                  constraints_list))
        return (constraints_resolution, constraints_dict,
                iters_convergence_dict, average_diffs_dict)

    def _create_ds_for_resolution_entity(self, sample, entity, variable_names,
                                         variables_count, variables_cats,
                                         sample_geo_names):
        ipf_ds_geo = IPF_DS(sample, entity, variable_names,
                            variables_count, variables_cats,
                            sample_geo_names)
        seed_geo = ipf_ds_geo.get_seed()

        ipf_ds_all = IPF_DS(sample, entity, variable_names,
                            variables_count, variables_cats)

        seed_all = ipf_ds_all.get_seed()
        row_idx = ipf_ds_all.get_row_idx(seed_all)
        return (seed_geo, seed_all, row_idx)

    def _run_ipf_all_geos(self, entity, seed_geo, seed_all, row_idx, marginals,
                          variable_names, variables_count, variables_cats,
                          variables_cats_count, geo_ids, geo_corr_to_sample):
        ipf_results = pd.DataFrame(index=seed_all.index)
        ipf_iters_convergence = pd.DataFrame(index=["iterations"])
        ipf_avgerage_diffs = pd.DataFrame(index=["average_percent_deviation"])
        for geo_id in geo_ids:
            # print "\tIPF for Geo: %s for Entity: %s" % (geo_id, entity)
            sample_geo_id = geo_corr_to_sample.loc[geo_id,
                                                   self.sample_geo_name]
            if isinstance(sample_geo_id, pd.Series):
                seed_geo_levels_list = range(len(seed_geo.index.names))
                seed_for_geo_id = (seed_geo.loc[sample_geo_id.tolist()]
                                   .sum(level=seed_geo_levels_list[1:]))
                # print (seed_geo.loc[sample_geo_id.tolist()])
                # print sample_geo_id.tolist(), "Satisfied series check"
            # if sample_geo_id.shape[0] >= 1:
            elif sample_geo_id > 0:
                seed_for_geo_id = seed_geo.loc[sample_geo_id.tolist()]
                # print sample_geo_id.tolist(), "Satisfied valid value check"
            elif sample_geo_id == -1:
                seed_for_geo_id = seed_all.copy()
                # print sample_geo_id.tolist(), "Satisfied default value check"
            else:
                raise Exception("Not series nor is it default value of -1")
            # print seed_for_geo_id
            # print "This is the sample geo id"
            # raw_input()

            marginals_geo = marginals.loc[geo_id]
            ipf_obj_geo = IPF(seed_all, seed_for_geo_id, row_idx,
                              marginals_geo, self.ipf_config,
                              variable_names, variables_cats,
                              variables_cats_count)
            # ipf_obj_geo.correct_zero_cell_issue()
            # ipf_results_geo = ipf_obj_geo.run_ipf()
            ipf_obj_geo.run_ipf()
            ipf_results[geo_id] = ipf_obj_geo.frequencies
            ipf_iters_convergence[geo_id] = (
                ipf_obj_geo.iter_convergence)
            ipf_avgerage_diffs[geo_id] = (
                ipf_obj_geo.average_diff_iters[-1])
            # print ('\t', ipf_obj_geo.iter_convergence,
            #         ipf_obj_geo.average_diff_iters)
            if (ipf_results[geo_id] == 0).any():
                raise Exception("""IPF cell values of zero are returned. """
                                """Needs troubleshooting""")

            # ipf_results[geo_id] = ipf_results_geo["frequency"]
            # raw_input("IPF for Geo: %s for Entity: %s complete"
            #          % (geo_id, self.entity))
        # print ipf_iters_convergence.T
        # print ipf_avgerage_diffs.T
        # raw_input("IPF Results")
        return (ipf_results, ipf_iters_convergence.T, ipf_avgerage_diffs.T)

    def _get_stacked_constraints(self, constraints_list):
        if len(constraints_list) == 0:
            return None
        elif len(constraints_list) == 1:
            return constraints_list[0].T
        stacked_constraints = constraints_list[0].T
        for constraint in constraints_list[1:]:
            len_left_frame_index = len(stacked_constraints.columns.values[0])
            len_right_frame_index = len(constraint.T.columns.values[0])

            if len_left_frame_index >= len_right_frame_index:
                stacked_constraints = stacked_constraints.join(constraint.T)
            else:
                stacked_constraints = constraint.T.join(stacked_constraints)
        stacked_constraints.sort_index(axis=1, inplace=True)

        stacked_constraints.columns = pd.Index(stacked_constraints.columns,
                                               tuplelize_cols=False)
        return stacked_constraints

    def _get_columns_constraints_dict(self, constraints_dict):
        columns_constraints_dict = {}
        for entity, constraints in constraints_dict.iteritems():
            columns_constraints_dict[entity] = (constraints
                                                .index.values.tolist())
        # print columns_constraints_dict
        return columns_constraints_dict

    def _get_frequencies_for_resolution(self, geo_ids, constraints_dict,
                                        procedure="bucket"):
        # TODO: Implemente other procedures for integerizing multiway freq
        frequencies_list = []

        for entity in self.housing_entities:
            print ("\tRounding frequencies for Entity: %s complete" % entity)

            frequencies = constraints_dict[entity].copy()

            for geo_id in geo_ids:
                # print ("Rounding Frequencies for Geo: %s for Entity: %s"
                #       % (geo_id, entity))

                frequency_geo = frequencies.loc[:, geo_id].values
                adjusted_frequency_geo = []
                accumulated_difference = 0

                for frequency in frequency_geo:
                    frequency_int = np.floor(frequency)
                    frequency_dec = frequency - frequency_int
                    accumulated_difference += frequency_dec
                    adjustment = accumulated_difference.round()
                    adjusted_frequency_geo.append(frequency_int + adjustment)
                    accumulated_difference -= adjustment
                frequencies.loc[:, geo_id] = adjusted_frequency_geo

            frequencies_list.append(frequencies)
        frequencies_resolution = (self._get_stacked_constraints(
                                  frequencies_list))
        return frequencies_resolution
