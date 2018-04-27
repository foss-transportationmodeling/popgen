import copy
import time

import numpy as np
import pandas as pd

from reweighting import Run_Reweighting
from config import ConfigError


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
                 adjust_household_filter_based_on_puma):
        self.seed_all = seed_all
        self.seed = seed
        self.idx = idx
        self.marginals = marginals
        self.ipf_config = ipf_config
        self.variable_names = variable_names
        self.variables_cats = variables_cats
        self.variables_cats_count = variables_cats_count
        self.adjust_household_filter_based_on_puma = \
            adjust_household_filter_based_on_puma
        self.ipf_iters = self.ipf_config.iterations
        self.ipf_tolerance = self.ipf_config.tolerance
        self.zero_marginal_correction = (
            self.ipf_config.zero_marginal_correction)
        self.archive_performance_frequency = (
            self.ipf_config.archive_performance_frequency)
        self.average_diff_iters = []
        self.average_diff_archive = []

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

            (converged, average_diff) = self._check_convergence()

            if ((self.archive_performance_frequency > 0 )&
                ((c_iter % self.archive_performance_frequency) == 0)):
                self.average_diff_archive.append(average_diff)

            if converged:
                # print "\t\t\tConvergence achieved in %d iter" % (c_iter)
                self.iter_convergence = c_iter
                break

    def _correct_zero_cell_issue(self):
        # print self.seed_all.index.difference(self.seed.index)

        if self.seed.shape[0] != self.seed_all.shape[0]:
            self.seed_all["prob"] = (self.seed["frequency"] /
                                     self.seed["frequency"].sum())

            if self.adjust_household_filter_based_on_puma == 1:
                self.seed_all.fillna(0., inplace=True)
            else:
                null_rows = self.seed_all["prob"].isnull()
                self.seed_all["prob_all"] = (self.seed_all["frequency"] /
                                             self.seed_all["frequency"].sum())
                self.seed_all.loc[null_rows, "prob"] = (
                    self.seed_all.loc[null_rows,    "prob_all"])
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
                marginal = self.marginals.loc[(var, cat)]
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
                return True, average_diff
        return False, average_diff

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
                original_frequency = self.marginals.loc[(var, cat)]
                if original_frequency == 0:
                    original_frequency = self.zero_marginal_correction
            diff_sum += (np.abs(adjusted_frequency - original_frequency) /
                         original_frequency)
        average_diff = diff_sum/self.variables_cats_count
        # print "Average Deviation", average_diff
        return average_diff


class Run_IPF(object):
    def __init__(self, entities, housing_entities, person_entities,
                 column_names_config, scenario_config, db):
        self.entities = entities
        self.housing_entities = housing_entities
        self.person_entities = person_entities
        self.column_names_config = column_names_config
        self.scenario_config = scenario_config
        self.db = db
        self.ipf_config = self.scenario_config.parameters.ipf
        try:
            self.adjust_household_ipf_config = \
                self.ipf_config.adjust_household_ipf_wrt_person_total
        except ConfigError, e:
            print e
            self.adjust_household_ipf_config = None

        try:
            self.adjust_household_filter_based_on_puma = \
                self.adjust_household_ipf_config.filter_based_on_puma
        except Exception, e:
            print e
            self.adjust_household_filter_based_on_puma = 0

        self.ipf_rounding = self.ipf_config.rounding_procedure
        self.sample_geo_names = self.column_names_config.sample_geo

        if self.sample_geo_names is None:
            pass
        elif isinstance(self.sample_geo_names, str):
            self.sample_geo_names = [self.sample_geo_names]
        else:
            self.sample_geo_names = \
                self.sample_geo_names.return_list()

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

        # self.geo_constraints.to_csv("C:\\Users\\kkonduri\\Google Drive\\misc\\HW_1_bmc_taz_2012_input_files\\geo_constraints.csv")
        # raw_input()

        if self.adjust_household_ipf_config is not None:
            self.geo_constraints_adjusted_household = \
                self._run_weighting_to_adjust_household_ipf_wrt_person_total()
            # print "Sum before adjusting"
            # print self.geo_constraints.sum(axis=0).sum()

            match_housing_total = \
                self.adjust_household_ipf_config.match_housing_total

            # print "Sum after adjusting"
            # print self.geo_constraints_adjusted_household.sum(axis=0).sum()

            # print match_housing_total, "-------"

            if match_housing_total == 1:
                self._fix_geo_constraints_to_match_housing_total(geo_ids)

            # print "Sum after matching"
            # print self.geo_constraints_adjusted_household.sum(axis=0).sum()

            # raw_input()

        if self.ipf_rounding == "bucket":
            self.geo_frequencies = (self._get_frequencies_for_resolution(
                                    geo_ids, "bucket"))


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
            print "\t\t\t", variable_names
            print "\t\t\t", variables_cats
            variables_count = len(variable_names)
            variables_cats_count = sum([len(cats) for cats in
                                        variables_cats.values()])
            (seed_geo,
             seed_all,
             row_idx) = (self._create_ds_for_resolution_entity(
                         sample, entity, variable_names,
                         variables_count, variables_cats))

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
                                         variables_count, variables_cats):
        seed_geo = {}
        for sample_geo_name in self.sample_geo_names:
            ipf_ds_geo = IPF_DS(sample, entity, variable_names,
                                variables_count, variables_cats,
                                sample_geo_name)
            seed_geo[sample_geo_name] = ipf_ds_geo.get_seed()

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
                                                   self.sample_geo_names]
            # print "\nHere are the sample geo ids"
            # print sample_geo_id

            seed_for_geo_id_all_sample_names = 0
            for index, sample_geo_name in enumerate(self.sample_geo_names):
                sample_geo_id_for_name = sample_geo_id[sample_geo_name]
                seed_geo_for_name = seed_geo[sample_geo_name]
                # print "\nfor sample_geo_name: {0}".format(sample_geo_name)
                # print "corresponding ids:", sample_geo_id_for_name, type(sample_geo_id_for_name)

                if isinstance(sample_geo_id_for_name, pd.Series):
                    # print "Inside series"
                    geo_id_for_name_filter = sample_geo_id_for_name > 0
                    sample_geo_id_for_name = sample_geo_id_for_name[
                        geo_id_for_name_filter]
                    seed_geo_levels_list = range(len(seed_geo_for_name.index.names))
                    seed_for_geo_id_for_name = (seed_geo_for_name.loc[sample_geo_id_for_name.tolist()]
                                       .sum(level=seed_geo_levels_list[1:]))
                elif sample_geo_id_for_name > 0:
                    seed_for_geo_id_for_name = seed_geo_for_name.loc[sample_geo_id_for_name.tolist()]
                elif sample_geo_id_for_name == -1:
                    seed_for_geo_id_for_name = seed_all.copy()
                else:
                    raise Exception("Not series nor is it default value of -1")

                if index == 0:
                    seed_for_geo_id_all_sample_names = seed_for_geo_id_for_name[["frequency"]]
                elif index > 0:
                    seed_for_geo_id_all_sample_names = \
                        seed_for_geo_id_all_sample_names.add(seed_for_geo_id_for_name[["frequency"]], fill_value=0)

                # print seed_for_geo_id_for_name.head()
                # print "------------------------------------------"
            # print seed_for_geo_id_all_sample_names.head()
            seed_for_geo_id_all_sample_names.reset_index(inplace=True)
            groupby_columns = ["entity"] + variable_names
            seed_for_geo_id_all_sample_names.set_index(groupby_columns, inplace=True)
            # print "_______"
            # print "after reseeding"
            # print seed_for_geo_id_all_sample_names.head()
            # print seed_for_geo_id_all_sample_names.shape
            # print seed_all.shape

            """
            # Old Implementation
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
            """

            marginals_geo = marginals.loc[geo_id]
            ipf_obj_geo = IPF(seed_all, seed_for_geo_id_all_sample_names, row_idx,
                              marginals_geo, self.ipf_config,
                              variable_names, variables_cats,
                              variables_cats_count,
                              self.adjust_household_filter_based_on_puma)
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
            # print ipf_obj_geo.frequencies.shape
            # raw_input("IPF for Geo: %s for Entity: %s complete"
            #          % (geo_id, entity))
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


    def _prepare_geo_constraints_to_adjust_household_ipf(self):
        housing_constraints_list = []
        for entity, constraint in self.geo_constraints_dict.iteritems():
            if entity in self.housing_entities:
                housing_constraints_list.append(constraint)

        self.housing_geo_constraints = self._get_stacked_constraints(
            housing_constraints_list)

        person_constraints_list = []
        for entity, constraint in self.geo_constraints_dict.iteritems():
            if entity in self.person_entities:
                person_constraints_list.append(constraint)

        self.person_geo_constraints = self._get_stacked_constraints(
            person_constraints_list)

        for index, entity in enumerate(self.person_entities):
            # print self.db.geo_marginals[entity].columns
            person_total_marginal_variable = \
                self.adjust_household_ipf_config.person_total_marginal_variable
            if index == 0:
                person_total = self.db.geo_marginals[entity].loc[
                    :, [person_total_marginal_variable]]
            else:
                person_total = person_total + \
                    self.db.geo_marginals[entity].loc[:, [person_total_marginal_variable]]

        # print person_total.head()
        new_columns = person_total.columns.droplevel(level=1)
        # print new_columns
        person_total.columns = new_columns

        # print person_total.head()
        self.housing_geo_constraints = self.housing_geo_constraints.join(person_total)

        # print housing_geo_constraints.head()
        # print housing_geo_constraints.columns
        # raw_input()
        # return housing_geo_constraints

    def _run_weighting_to_adjust_household_ipf_wrt_person_total(self):
        print "Adjusting household type frequencies to correct for person total ... "
        t = time.time()

        self._prepare_geo_constraints_to_adjust_household_ipf()

        # geo_controls_config = self.scenario_config.control_variables.geo
        control_variables_config = self.scenario_config.control_variables
        # reweighting_config = \
        #    self.scenario_config.parameters.ipf.adjust_household_ipf_wrt_person_total
        self.run_reweighting_obj = Run_Reweighting(
            self.entities, self.housing_entities, self.person_entities,
            self.column_names_config,
            self.adjust_household_ipf_config,
            control_variables_config, self.db,
            adjust_household_ipf_config=self.adjust_household_ipf_config)
        self.run_reweighting_obj.create_ds_for_adjust_household()
        # region_constraints are being set to zero because in this part,
        # we do not need to consider them

        self.run_reweighting_obj.run_reweighting_adjust_household(
            geo_constraints=self.housing_geo_constraints,
            filter_based_on_puma=self.adjust_household_filter_based_on_puma)

        housing_geo_constraints_adjusted_household = \
            self.run_reweighting_obj.region_sample_weights.fillna(0)
        housing_geo_constraints_adjusted_household = \
            housing_geo_constraints_adjusted_household.transpose()
        housing_geo_constraints_adjusted_household.columns = \
            pd.Index(housing_geo_constraints_adjusted_household.columns,
                     tuplelize_cols=False)
        geo_constraints_adjusted_household = \
            housing_geo_constraints_adjusted_household.join(
                self.person_geo_constraints)
        # print geo_constraints_adjusted_household
        # print "Weighting for adjusting household frequencies completed in: %.4f" % (time.time() - t)
        # raw_input()
        return geo_constraints_adjusted_household

    def _parse_geo_constraints_columns_by_entities(self, geo_constraints):
        columns_dict = {}

        for column in geo_constraints.columns.tolist():
            # print column

            if self.adjust_household_ipf_config is None:
                entity = column[0]
            else:
                if isinstance(column[0], tuple):
                    entity = column[0][0]
                else:
                    entity = column[0]
            if entity in columns_dict.keys():
                columns_dict[entity].append(column)
            else:
                columns_dict[entity] = [column]
            # print "\t", entity
        return columns_dict

    def _fix_geo_constraints_to_match_housing_total(self, geo_ids):
        columns_dict_original = self._parse_geo_constraints_columns_by_entities(
            self.geo_constraints)

        columns_dict_adjusted = self._parse_geo_constraints_columns_by_entities(
            self.geo_constraints_adjusted_household)

        for geo_id in geo_ids:
            for entity in self.housing_entities:
                total_housing_original = \
                    self.geo_constraints.loc[
                        geo_id, columns_dict_original[entity]].sum()

                total_housing_adjusted = \
                    self.geo_constraints_adjusted_household.loc[
                        geo_id, columns_dict_adjusted[entity]].sum()

                # print "Total original: {0} and adjusted: {1} for entity: {2}".format(
                #    total_housing_original, total_housing_adjusted, entity)

                if total_housing_adjusted > 0:
                    correction = total_housing_original / total_housing_adjusted
                    self.geo_constraints_adjusted_household.loc[
                        geo_id, columns_dict_adjusted[entity]] *= correction
                # else:
                #    raw_input("Don't match here: {0} with original being:{1}".format(total_housing_adjusted, total_housing_original))

    def _get_frequencies_for_resolution(self, geo_ids, procedure="bucket"):
        # TODO: Implemente other procedures for integerizing multiway freq
        frequencies_list = []

        if self.adjust_household_ipf_config is None:
            geo_constraints = self.geo_constraints.copy()
        else:
            geo_constraints = \
                self.geo_constraints_adjusted_household.copy()

        columns_dict = self._parse_geo_constraints_columns_by_entities(
            geo_constraints)

        for entity in self.housing_entities:
            # print ("\tRounding frequencies for Entity: %s complete" % entity)

            columns_for_entity = columns_dict[entity]
            frequencies = geo_constraints[columns_for_entity].T
            # print ("Working on entity: {0} for columns: {1}".format(
            #    entity, columns_for_entity))
            # print frequencies
            # print "Count before adjusting:{0}".format(frequencies.sum().sum())

            for geo_id in geo_ids:
                # print ("Rounding Frequencies for Geo: %s for Entity: %s"
                #       % (geo_id, entity))

                frequency_geo = frequencies.loc[:, geo_id].values
                adjusted_frequency_geo = []
                accumulated_difference = 0

                for frequency in frequency_geo:

                    # This is to avoid invalid constraints from getting
                    # a valid frequency
                    # if frequency == 0:
                    #    adjusted_frequency_geo.append(frequency)
                    #    continue
                    frequency_int = np.floor(frequency)
                    frequency_dec = frequency - frequency_int
                    accumulated_difference += frequency_dec
                    adjustment = accumulated_difference.round()
                    adjusted_frequency_geo.append(frequency_int + adjustment)
                    accumulated_difference -= adjustment
                frequencies.loc[:, geo_id] = adjusted_frequency_geo

            frequencies_list.append(frequencies)
            # print "Count after adjusting:{0}".format(frequencies.sum().sum())

        frequencies_resolution = (self._get_stacked_constraints(
                                  frequencies_list))
        return frequencies_resolution
