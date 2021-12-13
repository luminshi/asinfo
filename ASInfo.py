# asinfo produces a Pandas DataFrame (df) that contains merged public AS-level info on the Internet.

import os
import json
import re
import sys
import gzip
import ipaddress
from functools import lru_cache
from collections import defaultdict

import pandas as pd
import numpy as np
from SubnetTree import SubnetTree


class ASInfo:
    def __init__(self, dataset_path="datasets/"):
        # The root dataset path.
        self.DATASET_PATH = dataset_path
        CHECK_DATASET_PATH = os.path.isdir(self.DATASET_PATH)
        # If folder doesn't exist, then create it.
        if not CHECK_DATASET_PATH:
            os.makedirs(self.DATASET_PATH)
        
        # The paths of datasets.
        # We will rework the following dataset path settings
        # so we can always fetch the latest dataset at the runtime.
        
        # The AS relationship (CAIDA) dataset path
        # Src: https://publicdata.caida.org/datasets/as-relationships
        self.as_relationship_dataset = self.DATASET_PATH + "/caida/" + "20211001.as-rel.txt.bz2"
        
        # The Prefix to ASN  (CAIDA) dataset path
        # Src: http://data.caida.org/datasets/routing/routeviews-prefix2as/
        self.prefix_to_asn = self.DATASET_PATH + "/caida/" + "routeviews-rv2-20211018-1200.pfx2as.gz"
        
        # The AS to Organization (CAIDA) dataset path
        # Src: https://publicdata.caida.org/datasets/as-organizations
        self.asn_to_org = self.DATASET_PATH + "/caida/" + "20211001.as-org2info.jsonl.gz"

        # The PeeringDB nets dataset path
        # How to download the entire peeringdb dump:
        # `curl "https://www.peeringdb.com/api/net" -o all_some_date.json`
        self.peeringdb_nets = self.DATASET_PATH + "/peeringdb/nets/" + "all_20211019.json"
        
        # Initialize empty df variables
        self.as_relationship_df = None
        self.as_basic_df = None
        self.prefix_to_asn_df = None
        self.as_to_org_df = None
        self.peeringdb_net_df = None
        
        # This variable caches merged df using all dfs above
        self.all_df = None
        
        # Initialize SubnetTree for IP lookup
        self.subnet_tree = None
        
        # Build the base dfs based on the datasets above.
        self.__build()
    
    def save_asinfo_to_file(self, asinfo_path):
        # TODO
        pass
    
    def load_asinfo_from_file(self, asinfo_path):
        # TODO
        pass

    def merge(self, **kwargs):
        # the goal is to merge the df based on the user's preferences.
        # as_basic_df will be returned by default.
        result = self.as_basic_df
        merge_all = False

        if 'all' in kwargs and kwargs['all'] == True:
            merge_all = True
        
        if merge_all or ('as_to_org' in kwargs and kwargs['as_to_org'] == True):
            result = self.__merge_as_to_org(result)
        if merge_all or ('peeringdb_net' in kwargs and kwargs['peeringdb_net'] == True):
            result = self.__merge_peeringdb_net(result)
        if merge_all or ('asn_ip_count' in kwargs and kwargs['asn_ip_count'] == True):
            result = self.__merge_asn_ip_count(result)
        
        # fill NaA fields with empty string
        result = result.fillna("")

        if merge_all:
            self.all_df = result
            
        return result
    
    @lru_cache(maxsize=50000)
    def get_upstream_ases_by_asn(self, asn):
        df = self.as_relationship_df
        upstream_ases = df[(df['relation'] == -1) & (df['as1'] == asn)]
        upstream_ases = upstream_ases.merge(self.as_to_org_df, how='left', left_on='as1', right_on='asn')[
            ['as0', 'as1', 'relation', 'org_name', 'country']
        ]
        return upstream_ases
    
    @lru_cache(maxsize=50000)
    def get_customer_ases_by_asn(self, asn):
        df = self.as_relationship_df
        downstream_ases = df[(df['relation'] == -1) & (df['as0'] == asn)]
        downstream_ases = downstream_ases.merge(self.as_to_org_df, how='left', left_on='as1', right_on='asn')[
            ['as0', 'as1', 'relation', 'org_name', 'country']
        ]
        return downstream_ases
    
    @lru_cache(maxsize=50000)
    def get_as_info_by_ip(self, ip:str):
        if self.all_df is None:
            self.merge(all=True)
        result = None
        try:
            ip_obj = ipaddress.ip_address(ip)
            result = self.all_df[ self.all_df.asn.isin(self.subnet_tree[str(ip_obj)]) ]
        except ValueError:
            print("Input IP: {} is not valid".format(ip))
        return result

    @lru_cache(maxsize=50000)
    def get_as_info_by_prefix(self, prefix:str):
        if self.all_df is None:
            self.merge(all=True)
        result = None
        try:
            prefix_obj = ipaddress.ip_network(prefix)
            # why iter is necessary here? if the prefix is a /32, the hosts() return a LIST instead of a GENERATOR!.
            ip_obj = next(iter(prefix_obj.hosts()))
            result = self.all_df[ self.all_df.asn.isin(self.subnet_tree[str(ip_obj)]) ]
        except ValueError:
            print("Input prefix: {} is not valid".format(prefix))
        except TypeError:
            print(f"the problematic prefix: {prefix_obj}")
        return result
    
    @lru_cache(maxsize=50000)
    def get_as_info_by_asn(self, asn:int):
        if self.all_df is None:
            self.merge(all=True)
        return self.all_df[ self.all_df.asn == asn ]
    
    @lru_cache(maxsize=50000)
    def get_prefix_by_asn(self, asn:int):
        return self.prefix_to_asn_df[self.prefix_to_asn_df.asn.map(lambda x: asn in x)]
    
    @lru_cache(maxsize=50000)
    def get_as_info_by_org_name(self, name:str):
        if self.all_df is None:
            self.merge(all=True)
        return self.all_df[ self.all_df.org_name.str.contains(name, case=False) ]

    def __build(self):
        self.as_relationship_df = self.__build_as_relationship()
        self.as_basic_df = self.__build_as_basic()
        self.prefix_to_asn_df = self.__build_prefix_to_asn()
        # can only build the subnet_tree after __build_prefix_to_asn()
        self.subnet_tree = self.__build_subnet_tree()
        self.as_to_org_df = self.__build_as_to_org()
        self.peeringdb_net_df = self.__build_peeringdb_net()
    
    def __merge_as_to_org(self, df_to_merge_with):
        return df_to_merge_with.merge(
                self.as_to_org_df[['asn', 'country', 'org_name']],
                on='asn', how="inner")

    def __merge_peeringdb_net(self, df_to_merge_with):
        """
        result = df_to_merge_with.merge(self.peeringdb_net_df[['asn', 'info_type', 'info_traffic',
                                       'info_ratio', 'info_scope', 'website']], on='asn', how="left")
        """
        result = df_to_merge_with.merge(self.peeringdb_net_df[['asn', 'info_type', 'info_traffic',
                                       'info_ratio', 'info_scope']], on='asn', how="left")
        
        # get ASes with traffic information
        ases_with_traffic_level_info = result[result['info_traffic'].str.len() > 1]
        # extract the traffic information with regex
        ases_with_traffic_level_info = ases_with_traffic_level_info['info_traffic'].str.extract(r'(\d+)-(\d+)([a-zA-Z]+)').rename(columns={0:"bw_low", 1:"bw_high", 2:"bw_factor"}).dropna()

        # Use Mbps as the base unit
        def get_bw_factor(unit):
            unit = unit.lower()
            if unit == "tbps":
                return 1000000
            elif unit == "gbps":
                return 1000
            elif unit == "mbps":
                return 1
            else:
                raise Exception(f"B.W. unit: {unit} is not handled")
        
        ases_with_traffic_level_info['bw_factor'] = ases_with_traffic_level_info['bw_factor'].apply(get_bw_factor)
        # convert data type
        ases_with_traffic_level_info = ases_with_traffic_level_info.astype({'bw_low': 'int', 'bw_high': 'int', 'bw_factor': 'int'})
        ases_with_traffic_level_info['bw_low_mbps'] = ases_with_traffic_level_info['bw_low'] * ases_with_traffic_level_info['bw_factor']
        ases_with_traffic_level_info['bw_high_mbps'] = ases_with_traffic_level_info['bw_high'] * ases_with_traffic_level_info['bw_factor']
        result['bw_low_mbps'] = ases_with_traffic_level_info['bw_low_mbps']
        result['bw_high_mbps'] = ases_with_traffic_level_info['bw_high_mbps']
        
        result.bw_low_mbps.fillna(-1, inplace=True)
        result.bw_high_mbps.fillna(-1, inplace=True)
        result.bw_low_mbps.replace(r'^\s*$', -1, regex=True, inplace=True)
        result.bw_high_mbps.replace(r'^\s*$', -1, regex=True, inplace=True)
        
        #result.astype({"bw_low_mbps": 'int', "bw_high_mbps": 'int'})
        
        # we can now drop the info_traffic column
        result.drop(columns=['info_traffic'])
        
        return result
    
    def __merge_asn_ip_count(self, df_to_merge_with):
        # build a df that maps num_ips to asn
        asn_to_ip_num = defaultdict(int)
        for record in self.prefix_to_asn_df.to_records(index=False):
            asn_list = record[2]
            # every asn gets some dup. ip nums!
            for asn in asn_list:
                # all prefixes should be in IPv4
                asn_to_ip_num[int(asn)] += 2 ** (32 - record[1])
        t = pd.DataFrame(asn_to_ip_num.items(), columns = ['asn', 'num_ips'])
        t = t.astype({'asn':'int64'})
        t = t.sort_values(by=['asn']).reset_index().drop(columns=['index'])

        # merge with the input df
        merged_df = df_to_merge_with.merge(t, on='asn', how='left')
        # make sure the number of ips column is int type
        merged_df['num_ips'] = merged_df['num_ips'].fillna(0)
        merged_df = merged_df.astype({'num_ips':'int64'})

        return merged_df

    def __build_peeringdb_net(self):
        peeringdb_dataset = json.load(open(self.peeringdb_nets))
        peeringdb_net_df = pd.DataFrame(peeringdb_dataset["data"])
        return peeringdb_net_df
    
    def __build_as_to_org(self):
        with gzip.open(self.asn_to_org, 'rt') as f:
            raw_asn_to_org_result = [json.loads(jline) for jline in f.read().splitlines()]
        asn_rows = [row for row in raw_asn_to_org_result if row['type'] == 'ASN']
        org_rows = [row for row in raw_asn_to_org_result if row['type'] == 'Organization']
        asn_df = pd.DataFrame(asn_rows)
        org_df = pd.DataFrame(org_rows)
        as_to_org_df = asn_df.merge(org_df, on='organizationId')[['asn', 'country', 'name_y']].rename(columns={'name_y': 'org_name'}).astype({'asn': 'int64'})
        return as_to_org_df
    
    def __build_subnet_tree(self):
        t = SubnetTree()
        for e in self.prefix_to_asn_df.values.tolist():
            t[e[0] + "/" + str(e[1])] = e[2]
        return t
    
    def __build_prefix_to_asn(self):
        # Note that the original dataset format in gzip format.
        # We first load the dataset as a df
        prefix2as_df = pd.read_csv(self.prefix_to_asn, names=['prefix', 'prefix_len', 'asn'],
                           delim_whitespace=True, header=None, compression="gzip")

        """
        # Build filters to find rows with MOAS and as_set
        # A MOAS conflict occurs when a particular prefix appears to originate from more than one AS. 
        prefix2as_asn_filter_moas = prefix2as_df['asn'].str.contains("_")
        # Let's ignore AS set records for now; they are rare events.
        prefix2as_asn_filter_as_set = prefix2as_df['asn'].str.contains(",")
        """
        
        """
        Note on Multi-origin ASes (MOASes)
        (https://www.caida.org/catalog/datasets/routeviews-prefix2as/)
        The file format changed slightly beginning with the 2010-10-27 prefix-to-AS file.
        Previously, ASes were listed in sorted order in MOASes (for example, we said "10_20_30" and never "30_10_20").
        Now, we sort the ASes according to their frequency of appearance as an origin AS in the source BGP table.
        For example, suppose 10.0.0.0/8 is advertised by the ASes 10, 20, and 30,
        and suppose 7 RouteViews peers saw AS 10 as the origin AS, 4 saw AS 20 as the origin AS, and 29 saw AS 30 as the origin AS.
        Then the MOAS recorded in the prefix-to-AS file will be 30_10_20, according to descending frequency of appearance as an origin AS.
        If there is a tie in frequency, then we sort by lexicographical order. With the new sorting order,
        users who wish to choose "the best" mapping for simplicity (with full understanding of the caveats) can simply pick the first listed AS.
        """

        # Remove as_set from the prefix2as_df
        prefix2as_df = prefix2as_df[~prefix2as_df['asn'].str.contains(",")]
        prefix2as_df_copy = prefix2as_df.copy()
        # split MOAS ASNs into a list
        prefix2as_df_copy['asn'] = prefix2as_df.asn.str.split("_")
        prefix2as_df = prefix2as_df_copy
        prefix2as_df.asn = prefix2as_df.asn.apply(lambda x: [int(asn) for asn in x])
        
        # populate the asn seen by most vantage points.
        prefix2as_df['asn_single'] = prefix2as_df.apply(lambda row: row.asn[0], axis=1)
                
        return prefix2as_df

    def __build_as_relationship(self):
        as_relationship_df = pd.read_csv(self.as_relationship_dataset,
                                         names=['as0', 'as1', 'relation'],
                                         comment="#", sep="|", header=None,
                                         compression='bz2')
        return as_relationship_df
    
    def __build_as_basic(self):
        # Note that the original dataset format in in bz2.
        # We first load the dataset as a df
        as_relationship_df = pd.read_csv(self.as_relationship_dataset,
                                         names=['as0', 'as1', 'relation'],
                                         comment="#", sep="|", header=None,
                                         compression='bz2')
        
        # A df of unique ASNs
        unique_asn_list = pd.unique(as_relationship_df[['as0', 'as1']].values.ravel('K'))
        unique_asn_list = np.sort(unique_asn_list)
        as_df = pd.DataFrame({"asn": unique_asn_list})
        
        # A provider to customer df
        p2c_df = as_relationship_df[as_relationship_df['relation'] == -1]
        # A peer to peer df
        p2p_df = as_relationship_df[as_relationship_df['relation'] == 0]
        
        # Sum the provider count of each AS
        provider_count_df = as_df.merge(p2c_df['as1'], left_on="asn", right_on="as1", how="outer")
        provider_count_df = provider_count_df.groupby("asn").count().reset_index().rename(columns={'as1': 'provider_count'})
        
        # Sum the customer count of each AS
        customer_count_df = as_df.merge(p2c_df['as0'], left_on="asn", right_on="as0", how="outer")
        customer_count_df = customer_count_df.groupby("asn").count().reset_index().rename(columns={'as0': 'customer_count'})
        
        # Sum the peer count of each AS
        # (prob. not very readable; deal with it.)
        peer_count_df = as_df.copy()
        peer_count_df['peer_count'] = 0
        t1 = p2p_df.groupby('as0').size()
        t2 = p2p_df.groupby('as1').size()
        t1 = t1.rename_axis('asn').reset_index(name='peer_count')
        t2 = t2.rename_axis('asn').reset_index(name='peer_count')
        t3 = t1.merge(t2, on='asn', how='outer').fillna(0)
        t3['peer_count'] = t3['peer_count_x'] + t3['peer_count_y']
        peer_count_df = peer_count_df.merge(t3[['asn', 'peer_count']], on='asn', how='outer').fillna(0)
        peer_count_df['peer_count'] = (peer_count_df['peer_count_x'] + peer_count_df['peer_count_y'])
        peer_count_df = peer_count_df[['asn', 'peer_count']].astype({'peer_count': 'int32'})
        
        # Sum the sibling count of each AS
        # (prob. not very readable; deal with it.)
        sibling_count_df = as_df.copy()
        def find_sibling_ases(asn):
            return pd.unique(
                p2c_df[p2c_df['as0'].isin(p2c_df[p2c_df['as1'] == asn]['as0'].values)]['as1'].values.ravel('K')
            )
        sibling_count_df['sibling_count'] = sibling_count_df['asn'].apply(lambda x: max(find_sibling_ases(x).size - 1, 0))
        
        
        # Merge all dfs above
        all_as_df = provider_count_df.merge(customer_count_df, on="asn", how="outer")
        # Continue to merge with peer and sibling count dfs
        all_as_df = all_as_df.merge(peer_count_df, on='asn', how="outer")
        all_as_df = all_as_df.merge(sibling_count_df, on='asn', how="outer")
        
        # And we are done here; return the final df.
        return all_as_df


if __name__ == "__main__":
    asinfo = ASInfo()
    print(asinfo.merge(all=True).head(15))

