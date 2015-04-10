import pandas as pd
import numpy as np
from scipy import stats


class Draw_Population(object):
    def __init__(self, scenario_config, syn_population):
        self.scenario_config = scenario_config
        self.syn_population = syn_population

        self.geo_ids = self.syn_population.db.geo_ids
        self.geo_row_idx = self.syn_population.geo_row_idx
        self.geo_frequencies = self.syn_population.geo_frequencies
        self.geo_constraints = self.syn_population.geo_constraints
        self.geo_stacked = self.syn_population.geo_stacked
        self.region_sample_weights = self.syn_population.region_sample_weights

        self.iterations = (
            self.scenario_config.parameters.draws.iterations)
        self.seed = (
            self.scenario_config.parameters.draws.seed)
        self.pvalue_tolerance = (
            self.scenario_config.parameters.draws.pvalue_tolerance)

    def draw_population(self):
        np.random.seed(self.seed)
        print "Drawing Households"
        for geo_id in self.geo_ids:
            print "For geo:", geo_id
            geo_sample_weights = self.region_sample_weights.loc[:, geo_id]
            geo_cumulative_weights = (self._return_cumulative_probability(
                                      geo_sample_weights))
            geo_id_frequencies = self.geo_frequencies.loc[geo_id, :]
            geo_id_constraints = self.geo_constraints.loc[geo_id, :]

            p_value_max = -1
            for iter in range(self.iterations):
                #print "Iter is:", iter, self.iterations
                seed = self.seed + iter
                geo_id_rows_syn = self._pick_households(
                    geo_id_frequencies, geo_cumulative_weights)
                stat, p_value = self._measure_match(
                    geo_id_rows_syn, geo_id_constraints)
                if p_value > self.pvalue_tolerance:
                    (p_value_max, geo_id_rows_syn_max,
                     iter_max, stat_max, max_found) = (p_value,
                                                       geo_id_rows_syn, iter,
                                                       stat, True)
                    break
                elif p_value > p_value_max:
                    (p_value_max, geo_id_rows_syn_max,
                     iter_max, stat_max, max_found) = (p_value,
                                                       geo_id_rows_syn, iter,
                                                       stat, False)
            #print "Max found:", max_found, geo_id_frequencies.sum()
            print "Max iter: %d, %f, %f" % (iter_max, p_value_max, stat_max)
            self.syn_population.add_records_for_geo_id(
                geo_id, geo_id_rows_syn_max)

    def _return_cumulative_probability(self, geo_sample_weights):
        geo_cumulative_weights = {}

        for column in self.geo_frequencies.columns.values:
            rows = self.geo_row_idx[column]
            weights = geo_sample_weights.take(rows)
            geo_cumulative_weights[column] = (weights / weights.sum()).cumsum()

            #print geo_cumulative_weights[column]
        return geo_cumulative_weights

    def _pick_households(self, geo_id_frequencies, geo_cumulative_weights):
        last = 0
        rand_numbers = np.random.random(geo_id_frequencies.sum())
        list_rows_syn_subpop = []
        for column in self.geo_frequencies.columns.values:
            rows = self.geo_row_idx[column]
            column_frequency = geo_id_frequencies[column]
            column_bins = np.searchsorted(geo_cumulative_weights[column],
                                          (rand_numbers[
                                           last:last + column_frequency]),
                                          side="right")
            last += column_frequency
            list_rows_syn_subpop.append(rows.take(column_bins))
        geo_id_rows_syn = np.sort(np.concatenate(list_rows_syn_subpop))
        return geo_id_rows_syn

    def _measure_match(self, geo_id_rows_syn, geo_id_constraints,
                       over_columns=None):

        geo_id_synthetic = self.geo_stacked.take(geo_id_rows_syn).sum()
        geo_id_synthetic = pd.DataFrame(geo_id_synthetic,
                                        columns=["synthetic_count"])
        geo_id_synthetic["constraint"] = geo_id_constraints

        stat, p_value = stats.chisquare(geo_id_synthetic["synthetic_count"],
                                        geo_id_synthetic["constraint"])
        return stat, p_value
