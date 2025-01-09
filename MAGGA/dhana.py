from ryu.base import app_manager
from ryu.controller import mac_to_port
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.mac import haddr_to_bin
from ryu.lib.packet import packet
from ryu.lib.packet import arp
from ryu.lib.packet import ethernet
from ryu.lib.packet import ipv4
from ryu.lib.packet import ipv6
from ryu.lib.packet import ether_types
from ryu.lib import mac, ip
from ryu.topology.api import get_switch, get_link
from ryu.app.wsgi import ControllerBase
from ryu.topology import event
from collections import deque
from ryu.lib import hub
from threading import Lock


from collections import defaultdict
from operator import itemgetter

import os
import random
import time
import json
import networkx as nx  # Add networkx
import logging
import matplotlib.pyplot as plt
import numpy as np
import pickle
import subprocess


# Cisco Reference bandwidth = 1 Gbps
REFERENCE_BW = 10000000

DEFAULT_BW = 10000000

MAX_PATHS = 3

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



class ProjectController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    
    def load_host_values(self):
        def normalize(value, min_value, max_value):
            """Normalize a value between 0 and 1 based on the given min and max."""
            if max_value - min_value == 0:
                return 0
            return (value - min_value) / (max_value - min_value)

        with open('example.json') as f:
            data = json.load(f)
    
        min_asset_value = min(service['asset_value'] for service in data['service'])
        max_asset_value = max(service['asset_value'] for service in data['service'])
        min_downtime_cost_per_hour = min(service['downtime_cost_per_hour'] for service in data['service'])
        max_downtime_cost_per_hour = max(service['downtime_cost_per_hour'] for service in data['service'])
        min_sla = min(service['uptime_SLA_percentage'] for service in data['service'])
        max_sla = max(service['uptime_SLA_percentage'] for service in data['service'])
        min_rps = min(service['max_requests_per_second'] for service in data['service'])
        max_rps = max(service['max_requests_per_second'] for service in data['service'])
        min_mttr = min(service['MTTR_per_hour'] for service in data['service'])
        max_mttr = max(service['MTTR_per_hour'] for service in data['service'])

        # Weight factors for each metric
        asset_weight = 0.3
        downtime_weight = 0.25
        sla_weight = 0.2
        rps_weight = 0.15
        mttr_weight = 0.1

        host_values = {}
        for service in data['service']:
            normalized_asset_value = normalize(service['asset_value'], min_asset_value, max_asset_value)
            normalized_downtime_cost_per_hour = normalize(service['downtime_cost_per_hour'], min_downtime_cost_per_hour, max_downtime_cost_per_hour)
            normalized_sla = normalize(service['uptime_SLA_percentage'], min_sla, max_sla)
            normalized_rps = normalize(service['max_requests_per_second'], min_rps, max_rps)
            normalized_mttr = normalize(service['MTTR_per_hour'], min_mttr, max_mttr)

            composite_score = (
                asset_weight * normalized_asset_value +
                downtime_weight * normalized_downtime_cost_per_hour +
                sla_weight * normalized_sla +
                rps_weight * normalized_rps -
                mttr_weight * normalized_mttr
        )
        
            # Mapping the host IP to its composite score
            host_values[service['IP']] = composite_score
    
        print(host_values)  # This will print the composite scores for each host IP
        return host_values


    def __init__(self, *args, **kwargs):
        super(ProjectController, self).__init__(*args, **kwargs)
        self.lock = Lock()
        self.congested_count = 0
        self.congested_links_over_time = []
        self.recalculation_times = []
        self.utilization_data = defaultdict(list)
        self.path_changes = defaultdict(list)  # Store path change history per flow
        self.path_cache = {}  # Initialize path cache
        self.flow_stats = defaultdict(list)
        self.successful_packets = {}  # Format: {(src_ip, dst_ip): successful_packets}
        self.sent_packets = {}  # Format: {(src_ip, dst_ip): sent_packets}
        self.throughput_data = defaultdict(list)
        self.packet_delivery_ratio_data = defaultdict(list)
        self.host_values = self.load_host_values()
        self.monitor_thread = hub.spawn(self._monitor)
        self.chosen_paths = {}  # Format: { (src_ip, dst_ip): path }
        self.critical_switches = {}
        self.current_congested_link = None
        self.congested_links = set()  # Store multiple congested links
        self.original_paths = {}  # Dictionary to store original paths (src, dst) -> original path
        self.global_switch_values = {}
        self.mac_to_port = {}
        self.topology_api_app = self
        self.datapath_list = {}
        self.previous_stats = {}
        self.arp_table = {}
        self.switches = []
        self.hosts = {}
        self.hostsip = {}
        self.multipath_group_ids = {}
        self.group_ids = []
        self.adjacency = defaultdict(dict)
        self.bandwidths = defaultdict(lambda: defaultdict(lambda: DEFAULT_BW))
        self.graph = nx.Graph()  # Add a NetworkX graph to store the topology
          # Create and start the Mininet instance


    def reset_variables(self):
    
        self.congested_count = 0
        self.congested_links_over_time = []
        self.recalculation_times = []
        self.utilization_data = defaultdict(list)
        self.path_cache = {}  # Initialize path cache
        self.path_changes = defaultdict(list)  # Store path change history per flow
        self.flow_stats = defaultdict(list)
        self.successful_packets = {}  # Format: {(src_ip, dst_ip): successful_packets}
        self.sent_packets = {}  # Format: {(src_ip, dst_ip): sent_packets}
        self.throughput_data = defaultdict(list)
        self.packet_delivery_ratio_data = defaultdict(list)
        self.chosen_paths = {}  # Format: { (src_ip, dst_ip): path }
        self.critical_switches = {}
        self.current_congested_link = None
        self.congested_links = set()  # Store multiple congested links
        self.original_paths = {}  # Dictionary to store original paths (src, dst) -> original path
        self.global_switch_values = {}
        self.mac_to_port = {}
        self.topology_api_app = self
        self.datapath_list = {}
        self.previous_stats = {}
        self.arp_table = {}
        self.switches = []
        self.hosts = {}
        self.hostsip = {}
        self.multipath_group_ids = {}
        self.group_ids = []
        self.adjacency = defaultdict(dict)
        self.bandwidths = defaultdict(lambda: defaultdict(lambda: DEFAULT_BW))
        self.graph = nx.Graph()  # Add a NetworkX graph to store the topology
          # Create and start the Mininet instance




        
        
    def _monitor(self):
        
        while True:
            #print(f"Host IP mappings: {self.hostsip}")
            #print(f"Host MAC mappings: {self.hosts}")
            #print(f"Datapath mapping: {self.datapath_list}")
            #print(f"Chosen paths: {self.chosen_paths}")
            for dp in self.datapath_list.values():
                self._request_stats(dp)
                #logging.info("Request sent to datapath: %016x", dp.id)
            	
            hub.sleep(5)    

    def _request_stats(self, datapath):
        #self.logger.debug('send stats request: %016x', datapath.id)
        
        
        # Use the lock while modifying shared attributes
        with self.lock:
            ofproto = datapath.ofproto
            parser = datapath.ofproto_parser
            req = parser.OFPPortStatsRequest(datapath, 0, ofproto.OFPP_ANY)
            datapath.send_msg(req)

    
    def determine_threshold(self, host_values):
        scores = list(host_values.values())
        threshold_percentile = 50  # Set this to a desired percentile (e.g., 90th percentile)
        threshold = np.percentile(scores, threshold_percentile)
        return threshold

    def get_cached_paths(self, src, dst):
        # Check if the paths are cached
        key = (src, dst)
        if key in self.path_cache:
            return self.path_cache[key]
    
        # If not cached, calculate paths and store in cache
        paths = self.get_paths(src, dst)
        self.path_cache[key] = paths
        return paths

    def get_paths(self, src, dst):
        '''
        Get all paths from src to dst using DFS algorithm    
        '''
        if src == dst:
            return [[src]]
        paths = []
        stack = [(src, [src])]
        while stack:
            (node, path) = stack.pop()
            for next in set(self.adjacency[node].keys()) - set(path):
                if next is dst:
                    paths.append(path + [next])
                else:
                    stack.append((next, path + [next]))
        return paths
    
    def get_link_cost(self, s1, s2):
        '''
        Get the link cost between two switches 
        '''
        e1 = self.adjacency[s1][s2]
        e2 = self.adjacency[s2][s1]
        bl = min(self.bandwidths[s1][e1], self.bandwidths[s2][e2])
        ew = REFERENCE_BW/bl
        return ew

    def get_path_cost(self, path):
        '''
        Get the path cost
        '''
        cost = 0
        for i in range(len(path) - 1):
            cost += self.get_link_cost(path[i], path[i+1])
        return cost

    def update_graph(self):
        '''
        Update the NetworkX graph with the current topology (links and switches)
        '''
        self.graph.clear()
        for s1 in self.adjacency:
            for s2 in self.adjacency[s1]:
                self.graph.add_edge(s1, s2, weight=self.get_link_cost(s1, s2))

    def calculate_global_switch_values(self):
        '''
        Use betweenness centrality to calculate switch values
        '''
        self.update_graph()
        centrality = nx.betweenness_centrality(self.graph, weight='weight', normalized=True)
        
        # Store global switch values in a sorted order
        self.global_switch_values = sorted(centrality.items(), key=lambda item: item[1], reverse=True)
        
        print("Global Switch Values (based on centrality):")
        for switch, value in self.global_switch_values:
            print(f"Switch {switch}: Centrality = {value}")
        #self.save_metrics('graph.pkl', self.graph)
    
    def get_optimal_paths(self, src, dst, dst_ip):
        '''
        Get the most optimal paths, taking into account switch values.
        '''
        # Check if paths are already cached
        paths = self.get_cached_paths(src, dst)
        
        if not paths:
            # If no cached paths are found, calculate new paths
            paths = self.get_paths(src, dst)
    
            # Optionally cache the newly calculated paths
            self.path_cache[(src, dst)] = paths
        dst_host_value = self.host_values.get(dst_ip, 0)
        switch_values = self.global_switch_values
        high_value_threshold = self.determine_threshold(self.host_values)
    
        # Calculate critical switches
        critical_switches = {switch for switch, value in switch_values if value > (sum(value for _, value in switch_values) / len(switch_values))}
        
        self.critical_switches = critical_switches
        #self.save_metrics('critical_switches.pkl', self.critical_switches)
    
        def path_cost_with_incentive(path):
            base_cost = self.get_path_cost(path)
            critical_count = sum(1 for node in path if node in critical_switches)
            length = len(path)
            length_incentive = length * 1.0  # Provide positive reward for longer paths
            critical_penalty = critical_count * 10.0
            return base_cost - length_incentive + critical_penalty  # Use subtraction for incentive

    
        if dst_host_value > high_value_threshold:
            # For high-value hosts, prefer shortest paths
            paths = sorted(paths, key=lambda x: self.get_path_cost(x))
            print("BEST paths from ", src, " to ", dst, " : ", paths)
        else:
            # For less valuable hosts, use path_cost_with_penalty to consider longer paths
            paths = sorted(paths, key=path_cost_with_incentive)
            print("AVAILABLE paths from ", src, " to ", dst, " : ", paths)

        # Limit the number of paths to MAX_PATHS
        paths_count = min(len(paths), MAX_PATHS)
        return paths[:paths_count]

        

    def add_ports_to_paths(self, paths, first_port, last_port):
        '''
        Add the ports that connects the switches for all paths
        '''
        paths_p = []
        for path in paths:
            p = {}
            in_port = first_port
            for s1, s2 in zip(path[:-1], path[1:]):
                out_port = self.adjacency[s1][s2]
                p[s1] = (in_port, out_port)
                in_port = self.adjacency[s2][s1]
            p[path[-1]] = (in_port, last_port)
            paths_p.append(p)
        return paths_p

    def generate_openflow_gid(self):
        '''
        Returns a random OpenFlow group id
        '''
        n = random.randint(0, 2**32)
        while n in self.group_ids:
            n = random.randint(0, 2**32)
        return n


    def install_paths(self, src, first_port, dst, last_port, ip_src, ip_dst):
        computation_start = time.time()
        paths = self.get_optimal_paths(src, dst, ip_dst)
        pw = []
        for path in paths:
            pw.append(self.get_path_cost(path))
            print(path, "cost = ", pw[len(pw) - 1])
        sum_of_pw = sum(pw) * 1.0
        paths_with_ports = self.add_ports_to_paths(paths, first_port, last_port)
        switches_in_paths = set().union(*paths)
        
        #print(f"Path from {ip_src} to {ip_dst} over switches: {paths}")

        for node in switches_in_paths:

            dp = self.datapath_list[node]
            ofp = dp.ofproto
            ofp_parser = dp.ofproto_parser

            ports = defaultdict(list)
            actions = []
            i = 0

            for path in paths_with_ports:
                if node in path:
                    in_port = path[node][0]
                    out_port = path[node][1]
                    if (out_port, pw[i]) not in ports[in_port]:
                        ports[in_port].append((out_port, pw[i]))
                i += 1

            for in_port in ports:

                match_ip = ofp_parser.OFPMatch(
                    eth_type=0x0800, 
                    ipv4_src=ip_src, 
                    ipv4_dst=ip_dst
                )
                match_arp = ofp_parser.OFPMatch(
                    eth_type=0x0806, 
                    arp_spa=ip_src, 
                    arp_tpa=ip_dst
                )

                out_ports = ports[in_port]
                # print out_ports 

                if len(out_ports) > 1:
                    group_id = None
                    group_new = False

                    if (node, src, dst) not in self.multipath_group_ids:
                        group_new = True
                        self.multipath_group_ids[
                            node, src, dst] = self.generate_openflow_gid()
                    group_id = self.multipath_group_ids[node, src, dst]

                    buckets = []
                    # print "node at ",node," out ports : ",out_ports
                    for port, weight in out_ports:
                        bucket_weight = int(round((1 - weight/sum_of_pw) * 10))
                        bucket_action = [ofp_parser.OFPActionOutput(port)]
                        buckets.append(
                            ofp_parser.OFPBucket(
                                weight=bucket_weight,
                                watch_port=port,
                                watch_group=ofp.OFPG_ANY,
                                actions=bucket_action
                            )
                        )

                    if group_new:
                        req = ofp_parser.OFPGroupMod(
                            dp, ofp.OFPGC_ADD, ofp.OFPGT_SELECT, group_id,
                            buckets
                        )
                        dp.send_msg(req)
                    else:
                        req = ofp_parser.OFPGroupMod(
                            dp, ofp.OFPGC_MODIFY, ofp.OFPGT_SELECT,
                            group_id, buckets)
                        dp.send_msg(req)

                    actions = [ofp_parser.OFPActionGroup(group_id)]

                    self.add_flow(dp, 32768, match_ip, actions)
                    self.add_flow(dp, 1, match_arp, actions)

                elif len(out_ports) == 1:
                    actions = [ofp_parser.OFPActionOutput(out_ports[0][0])]

                    self.add_flow(dp, 32768, match_ip, actions)
                    self.add_flow(dp, 1, match_arp, actions)
        print("Path installation finished in ", time.time() - computation_start)
        return paths_with_ports[0][src][1]

    def add_flow(self, datapath, priority, match, actions, buffer_id=None):
        # print "Adding flow ", match, actions
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS,
                                             actions)]
        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, buffer_id=buffer_id,
                                    priority=priority, match=match,
                                    instructions=inst)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                    match=match, instructions=inst)
        datapath.send_msg(mod)

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def _switch_features_handler(self, ev):
        print("switch_features_handler is called")
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        # Drop all IPv6 packets
        match_ipv6 = parser.OFPMatch(eth_type=0x86DD)
        self.add_flow(datapath, 1, match_ipv6, [])

        # General catch-all match (for other traffic)
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)


    @set_ev_cls(ofp_event.EventOFPPortDescStatsReply, MAIN_DISPATCHER)
    def port_desc_stats_reply_handler(self, ev):
        switch = ev.msg.datapath
        for p in ev.msg.body:
            self.bandwidths[switch.id][p.port_no] = p.curr_speed

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)
        arp_pkt = pkt.get_protocol(arp.arp)
        ipv4_pkt = pkt.get_protocol(ipv4.ipv4)

        # avoid broadcast from LLDP
        if eth.ethertype == 35020:
            return

        if pkt.get_protocol(ipv6.ipv6):  # Drop the IPV6 Packets.
            match = parser.OFPMatch(eth_type=eth.ethertype)
            actions = []
            self.add_flow(datapath, 1, match, actions)
            return None

        dst = eth.dst
        src = eth.src
        dpid = datapath.id

        if src not in self.hosts:
            self.hosts[src] = (dpid, in_port)

        out_port = ofproto.OFPP_FLOOD

        if arp_pkt:
            src_ip = arp_pkt.src_ip
            dst_ip = arp_pkt.dst_ip
            if src_ip not in self.hostsip:
                self.hostsip[src_ip] = (dpid, in_port)
    
            # Store ARP information
            if arp_pkt.opcode == arp.ARP_REPLY:
                self.arp_table[src_ip] = src
            elif arp_pkt.opcode == arp.ARP_REQUEST and dst_ip in self.arp_table:
                self.arp_table[src_ip] = src
    
            # Calculate global switch values once all switches and hosts are discovered
            if len(self.switches) == len(self.datapath_list) and len(self.hosts) == 4:
                self.calculate_global_switch_values()
    
                # Install paths after global switch values are calculated
            if arp_pkt.opcode == arp.ARP_REPLY:
                h1 = self.hosts[src]
                h2 = self.hosts[dst]
                out_port = self.install_paths(h1[0], h1[1], h2[0], h2[1], src_ip, dst_ip)
                self.install_paths(h2[0], h2[1], h1[0], h1[1], dst_ip, src_ip)  # reverse
            elif arp_pkt.opcode == arp.ARP_REQUEST and dst_ip in self.arp_table:
                dst_mac = self.arp_table[dst_ip]
                h1 = self.hosts[src]
                h2 = self.hosts[dst_mac]
                out_port = self.install_paths(h1[0], h1[1], h2[0], h2[1], src_ip, dst_ip)
                self.install_paths(h2[0], h2[1], h1[0], h1[1], dst_ip, src_ip)  # reverse



                

        actions = [parser.OFPActionOutput(out_port)]

        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data

        out = parser.OFPPacketOut(
            datapath=datapath, buffer_id=msg.buffer_id, in_port=in_port,
            actions=actions, data=data)
        datapath.send_msg(out)

    @set_ev_cls(event.EventSwitchEnter)
    def switch_enter_handler(self, ev):
        switch = ev.switch.dp
        ofp_parser = switch.ofproto_parser

        if switch.id not in self.switches:
            self.switches.append(switch.id)
            self.datapath_list[switch.id] = switch

            # Request port/link descriptions, useful for obtaining bandwidth
            req = ofp_parser.OFPPortDescStatsRequest(switch)
            switch.send_msg(req)
            
        

    @set_ev_cls(event.EventSwitchLeave, MAIN_DISPATCHER)
    def switch_leave_handler(self, ev):
        print(ev)
        switch = ev.switch.dp.id
        if switch in self.switches:
            self.switches.remove(switch)
            del self.datapath_list[switch]
            del self.adjacency[switch]






    @set_ev_cls(event.EventLinkAdd, MAIN_DISPATCHER)
    def link_add_handler(self, ev):
        s1 = ev.link.src
        s2 = ev.link.dst
        self.adjacency[s1.dpid][s2.dpid] = s1.port_no
        self.adjacency[s2.dpid][s1.dpid] = s2.port_no
        # Add the link to the graph
        self.graph.add_edge(s1.dpid, s2.dpid, weight=self.get_link_cost(s1.dpid, s2.dpid))

    @set_ev_cls(event.EventLinkDelete, MAIN_DISPATCHER)
    def link_delete_handler(self, ev):
        s1 = ev.link.src
        s2 = ev.link.dst
        try:
            del self.adjacency[s1.dpid][s2.dpid]
            del self.adjacency[s2.dpid][s1.dpid]
            # Remove the link from the graph
            self.graph.remove_edge(s1.dpid, s2.dpid)
        except KeyError:
            pass 
            


    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def port_stats_reply_handler(self, ev):
        body = ev.msg.body
        datapath = ev.msg.datapath
        logging.info(f"Received stats for {datapath.id}")
        current_time = time.time()
        congestion_threshold = 0.8

        dpid = datapath.id

        # Ensure previous_stats is initialized for the datapath
        if dpid not in self.previous_stats:
            self.previous_stats[dpid] = {}

        for stat in body:
            port_no = stat.port_no
    
            # Check for the special port value for all ports
            if port_no == 4294967294:  # This is OFP_PORT_ALL
                continue  # Skip processing for this special port number
    
            # Initialize previous stats for this port if not done
            if port_no not in self.previous_stats[dpid]:
                # Initialize previous statistics for this port
                self.previous_stats[dpid][port_no] = {'tx_bytes': 0, 'rx_bytes': 0}

            current_tx_bytes = stat.tx_bytes
            current_rx_bytes = stat.rx_bytes
    
            # Retrieve previous statistics
            previous_tx_bytes = self.previous_stats[dpid][port_no]['tx_bytes']
            previous_rx_bytes = self.previous_stats[dpid][port_no]['rx_bytes']

            # Calculate transmitted and received bytes since the last stats retrieval
            transmitted_bytes = max(0, current_tx_bytes - previous_tx_bytes)
            received_bytes = max(0, current_rx_bytes - previous_rx_bytes)

            # Determine total bytes only if there was traffic
            total_bytes = transmitted_bytes + received_bytes if (transmitted_bytes + received_bytes) > 0 else 0

            # Debug prints for tracking values
            #print(f"Port {port_no}:")
            #print(f"  Current TX bytes: {current_tx_bytes}, Previous TX bytes: {previous_tx_bytes}, Transmitted bytes: {transmitted_bytes}")
            #print(f"  Current RX bytes: {current_rx_bytes}, Previous RX bytes: {previous_rx_bytes}, Received bytes: {received_bytes}")
            #print(f"  Total bytes: {total_bytes}")

            # Update the previous statistics for this port
            self.previous_stats[dpid][port_no] = {
                'tx_bytes': current_tx_bytes,
                'rx_bytes': current_rx_bytes
            }
            # Calculate throughput in bytes per second
            throughput = (transmitted_bytes + received_bytes) / (time.time() - current_time)
            self.throughput_data[datapath.id].append((current_time, throughput))
    
            #self.save_metrics('throughput_data.pkl', self.throughput_data)
    
            # Update total bytes transmitted for this datapath
            # Calculate port utilization (simplified using bytes transmitted/received)
            bandwidth_limit = self.bandwidths[dpid][port_no]  # Assuming bandwidths are set correctly
            utilization = total_bytes / bandwidth_limit if bandwidth_limit > 0 else 0
    
            self.utilization_data[(dpid, port_no)].append((current_time, utilization))
            #print(f"Total bytes = {total_bytes}")
    
            # Check if link is congested
            with self.lock:  # Ensure thread-safe access to congested_links
                if utilization > congestion_threshold:
                    #print("Congested!")
                    self.congested_count += 1
                    if (dpid, port_no) not in self.congested_links:
                        self.congested_links.add((dpid, port_no))  # Add to congested links set
                else:
                    if (dpid, port_no) in self.congested_links:
                        self.congested_links.remove((dpid, port_no))  # Remove from congested links set
                        self.congested_count -= 1

            # Update the previous statistics
            self.previous_stats[(dpid, port_no)] = {
                'tx_bytes': current_tx_bytes,
                'rx_bytes': current_rx_bytes
            }

        self.handle_all_congested_links()
        
        self.congested_links_over_time.append((current_time, self.congested_count))
        #self.save_all_data()


    def handle_all_congested_links(self):  
        if len(self.congested_links) == 0:
            return
        for congested_link in self.congested_links:
            self.handle_congested_link(congested_link)

    def handle_congested_link(self, congested_link):
        switch_id, port_no = congested_link
        #print(f"Handling congestion on Switch {switch_id}, Port {port_no}. Rerouting flows...")
    
        with self.lock:# Request flow statistics from the switch
            datapath = self.datapath_list[switch_id]
            parser = datapath.ofproto_parser
            self.current_congested_link = congested_link
            req = parser.OFPFlowStatsRequest(datapath, 0, datapath.ofproto.OFPTT_ALL, datapath.ofproto.OFPP_ANY, datapath.ofproto.OFPG_ANY)
            datapath.send_msg(req)
        

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def flow_stats_reply_handler(self, ev):
        body = ev.msg.body
        datapath = ev.msg.datapath
        
        # Safely access the current congested link
        with self.lock:
            congested_link = self.current_congested_link
    
        if congested_link is None:
            self.logger.warning("No congested link is set.")
            return
        
        # Get the parser from the datapath
        parser = datapath.ofproto_parser
    
        for stat in body:
        
            match = stat.match
            packet_count = stat.packet_count
            byte_count = stat.byte_count
            switch_id = datapath.id
            eth_type = match.get('eth_type', None)
        
            # Update switch flow stats
            current_time = time.time()
            if switch_id not in self.flow_stats:
                self.flow_stats[switch_id] = []
            self.flow_stats[switch_id].append((current_time, packet_count, byte_count))
            #self.save_all_data()
            
            if 'ipv4_src' not in stat.match or 'ipv4_dst' not in stat.match:
                continue  # Skip entries without valid IP addresses
            
            if eth_type == 2048:  # IPv4 (can be used to filter out IPv4 traffic only)
                src = match.get('ipv4_src', 'Unknown')
                dst = match.get('ipv4_dst', 'Unknown')
                key = (src, dst)
                #print(key)
            
            

            # Increment sent and successful packet counts
            if key not in self.sent_packets:
                self.sent_packets[key] = 0
                self.successful_packets[key] = 0
            
            self.sent_packets[key] += packet_count
            self.successful_packets[key] += packet_count  # Assuming all sent packets are successful initially
              
                
            if packet_count > 0:
                delivery_ratio = packet_count / (packet_count + random.randint(0, 10))  # Simulated loss
                self.packet_delivery_ratio_data[datapath.id].append((current_time, delivery_ratio))

            #self.save_metrics('packet_delivery_ratio.pkl', self.packet_delivery_ratio_data)  
            
            instructions = stat.instructions
            
            
            if not instructions or not instructions[0].actions:
                #self.logger.warning(f"No actions found for match={stat.match}")
                continue
    
            for action in stat.instructions[0].actions:
                if isinstance(action, datapath.ofproto_parser.OFPActionOutput):
                    port_no = action.port
                    bandwidth = self.bandwidths[datapath.id][port_no]  # Get bandwidth for the port
                    utilization = (byte_count / bandwidth) * 100  # Convert to percentage
                    
                    if utilization > 80:  # Threshold for congestion
                        #self.logger.warning(f"Congestion detected on link: src={src}, dst={dst}, utilization={utilization}%")
                        self.recalculate_paths_avoiding_congested_link(src, dst, congested_link)
                        break  # Exit the loop after handling congestion   
                        
                        
    def get_optimal_paths_with_congestion(self, src, dst, dst_ip, src_ip):
        '''
        Get the most optimal paths, taking into account switch values and congestion.
        '''

        # Check the cache first
        if (src_ip, dst_ip) in self.path_cache:
            all_paths = self.path_cache[(src_ip, dst_ip)]
        else:
            # If not cached, calculate new paths
            all_paths = self.get_paths(src, dst)
            self.path_cache[(src_ip, dst_ip)] = all_paths  # Cache the result

        #print(f"All paths from {src} to {dst}: {all_paths}")

        # Create a list to hold paths and their congested link counts
        paths_with_congestion_counts = []

        for path in all_paths:
            congestion_count = self.count_congested_links(path)
            
            paths_with_congestion_counts.append((path, congestion_count))

        # Sort paths based on the number of congested links (ascending)
        paths_with_congestion_counts.sort(key=lambda x: x[1])
        #print(paths_with_congestion_counts)

        # Select the path with the least congested links
        if paths_with_congestion_counts:
            best_path = paths_with_congestion_counts[0][0]
            current_time = time.time()

            if (src_ip, dst_ip) not in self.path_changes:
                self.path_changes[(src_ip, dst_ip)] = []
            self.path_changes[(src_ip, dst_ip)].append((current_time, best_path))

            # Store multiple unique paths for the same src-dst pair
            if (src_ip, dst_ip) not in self.chosen_paths:
                self.chosen_paths[(src_ip, dst_ip)] = set()  # Use a set to hold unique paths
        
            # Add the best_path as a tuple to ensure uniqueness
            self.chosen_paths[(src_ip, dst_ip)].add(tuple(best_path))  # Convert to tuple for uniqueness
        
            #self.save_all_data()
            #print(f"Best path selected (least congested links): {best_path}")
        else:
            #print(f"No valid paths found.")
            return

    # Destination host value and critical switches
        dst_host_value = self.host_values.get(dst_ip, 0)
        switch_values = self.global_switch_values
        high_value_threshold = self.determine_threshold(self.host_values)

        critical_switches = self.critical_switches

        # Define path cost with incentive for critical switches and length
        def path_cost_with_incentive(path):
            base_cost = self.get_path_cost(path)
            critical_count = sum(1 for node in path if node in critical_switches)
            length_incentive = len(path) * 1.0  # Provide positive reward for longer paths
            critical_penalty = critical_count * 10.0
            congestion_count = self.count_congested_links(path)
            return base_cost - length_incentive + critical_penalty + congestion_count # Use subtraction for incentive

    # Choose the paths based on host value
        if dst_host_value > high_value_threshold:
            # For high-value hosts, prefer paths with least congestion first, then shortest
            paths = sorted(paths_with_congestion_counts, key=lambda x: (x[1]))
            #paths = sorted(paths_with_congestion_counts, key=lambda x: (x[1], self.get_path_cost(x[0])))
            #print("BEST paths from ", src, " to ", dst, " : ", [p[0] for p in paths])
        else:
            # For less valuable hosts, prefer paths with fewer critical switches and least congestion
            paths = sorted(paths_with_congestion_counts, key=lambda x: (path_cost_with_incentive(x[0]), x[1]))
            #print("AVAILABLE paths from ", src, " to ", dst, " : ", [p[0] for p in paths])

        # Limit the number of paths to MAX_PATHS
        paths_count = min(len(paths), MAX_PATHS)
        return [p[0] for p in paths[:paths_count]]

                        
                            
    
    
    def reinstall_paths(self, src, first_port, dst, last_port, ip_src, ip_dst):
        computation_start = time.time()
        paths = self.get_optimal_paths_with_congestion(src, dst, ip_src, ip_dst)
        pw = []
        for path in paths:
            pw.append(self.get_path_cost(path))
            #print(path, "cost = ", pw[len(pw) - 1])
        sum_of_pw = sum(pw) * 1.0
        paths_with_ports = self.add_ports_to_paths(paths, first_port, last_port)
        switches_in_paths = set().union(*paths)
        
        #print(f"Path from {ip_src} to {ip_dst} over switches: {paths}")

        for node in switches_in_paths:

            dp = self.datapath_list[node]
            ofp = dp.ofproto
            ofp_parser = dp.ofproto_parser

            ports = defaultdict(list)
            actions = []
            i = 0

            for path in paths_with_ports:
                if node in path:
                    in_port = path[node][0]
                    out_port = path[node][1]
                    if (out_port, pw[i]) not in ports[in_port]:
                        ports[in_port].append((out_port, pw[i]))
                i += 1

            for in_port in ports:

                match_ip = ofp_parser.OFPMatch(
                    eth_type=0x0800, 
                    ipv4_src=ip_src, 
                    ipv4_dst=ip_dst
                )
                match_arp = ofp_parser.OFPMatch(
                    eth_type=0x0806, 
                    arp_spa=ip_src, 
                    arp_tpa=ip_dst
                )

                out_ports = ports[in_port]
                # print out_ports 

                if len(out_ports) > 1:
                    group_id = None
                    group_new = False

                    if (node, src, dst) not in self.multipath_group_ids:
                        group_new = True
                        self.multipath_group_ids[
                            node, src, dst] = self.generate_openflow_gid()
                    group_id = self.multipath_group_ids[node, src, dst]

                    buckets = []
                    # print "node at ",node," out ports : ",out_ports
                    for port, weight in out_ports:
                        bucket_weight = int(round((1 - weight/sum_of_pw) * 10))
                        bucket_action = [ofp_parser.OFPActionOutput(port)]
                        buckets.append(
                            ofp_parser.OFPBucket(
                                weight=bucket_weight,
                                watch_port=port,
                                watch_group=ofp.OFPG_ANY,
                                actions=bucket_action
                            )
                        )

                    if group_new:
                        req = ofp_parser.OFPGroupMod(
                            dp, ofp.OFPGC_ADD, ofp.OFPGT_SELECT, group_id,
                            buckets
                        )
                        dp.send_msg(req)
                    else:
                        req = ofp_parser.OFPGroupMod(
                            dp, ofp.OFPGC_MODIFY, ofp.OFPGT_SELECT,
                            group_id, buckets)
                        dp.send_msg(req)

                    actions = [ofp_parser.OFPActionGroup(group_id)]

                    self.add_flow(dp, 32768, match_ip, actions)
                    self.add_flow(dp, 1, match_arp, actions)

                elif len(out_ports) == 1:
                    actions = [ofp_parser.OFPActionOutput(out_ports[0][0])]

                    self.add_flow(dp, 32768, match_ip, actions)
                    self.add_flow(dp, 1, match_arp, actions)
        #print("Path REinstallation finished in ", time.time() - computation_start)
        return paths_with_ports[0][src][1]
             
            

    # Update the recalculate_paths function to use reinstall_paths
    def recalculate_paths_avoiding_congested_link(self, src_ip, dst_ip, congested_link):
        
        #if src_ip == '10.0.0.3' and dst_ip == '10.0.0.1' or dst_ip == '10.0.0.3' and src_ip == '10.0.0.1':
            #return
            
        switch_id, port_no = congested_link
        start_time = time.time()

        # Get source and destination switch IDs from self.hostsip
        if src_ip not in self.hostsip or dst_ip not in self.hostsip:
            #print(f"Error: Missing IP addresses in self.hostsip. src_ip={src_ip}, dst_ip={dst_ip}")
            return

        h1 = self.hostsip[src_ip]
        #print("SRC IP:")
        #print(src_ip)
        h2 = self.hostsip[dst_ip]
        #print("DST IP:")
        #print(dst_ip)


        #print(f"Reinstalling path from {h1}  to {h2} ")
        out_port = self.reinstall_paths(h1[0], h1[1], h2[0], h2[1], src_ip, dst_ip)

        #print(f"Reinstalling reverse path from {h2} to {h1}")
        self.reinstall_paths(h2[0], h2[1], h1[0], h1[1], dst_ip, src_ip)  # reverse
        end_time = time.time()
        recalculation_time = end_time - start_time
        self.recalculation_times.append(recalculation_time)
        self.get_average_recalculation_time(len(self.switches))
    
    
    
    def count_congested_links(self, path):
        """
        Count the number of congested links in the given path, considering both switch and port (dpid, port_no).
        Each congested link (switch-port pair) is counted only once, even if multiple instances are found.
        """
        count = 0
        congested_link_pairs = set()  # To track congested (dpid, port_no) pairs
    
        for i in range(len(path) - 1):
            current_switch = path[i]
            next_switch = path[i + 1]
    
            # Find the port linking current_switch to next_switch
            current_port = self.get_outgoing_port(current_switch, next_switch)
            next_port = self.get_incoming_port(next_switch, current_switch)
    
            # Check if either link is congested
            if (current_switch, current_port) in self.congested_links:
                congested_link_pairs.add((current_switch, current_port))
            if (next_switch, next_port) in self.congested_links:
                congested_link_pairs.add((next_switch, next_port))
    
        count = len(congested_link_pairs)  # Count unique congested links
        return count

        
    def get_outgoing_port(self, current_switch, next_switch):
        """
        Returns the outgoing port on current_switch that connects to next_switch.
        """
        if current_switch in self.adjacency and next_switch in self.adjacency[current_switch]:
            return self.adjacency[current_switch][next_switch]
        else:
            raise ValueError(f"No link between {current_switch} and {next_switch} in adjacency matrix")

    def get_incoming_port(self, next_switch, current_switch):
        """
        Returns the incoming port on next_switch that connects to current_switch.
        """
        if next_switch in self.adjacency and current_switch in self.adjacency[next_switch]:
            return self.adjacency[next_switch][current_switch]
        else:
            raise ValueError(f"No link between {next_switch} and {current_switch} in adjacency matrix")

   
    def get_average_recalculation_time(self, num_switches):
        if self.recalculation_times:
            avg_time = sum(self.recalculation_times) / len(self.recalculation_times)
            # Save the number of switches and the average time to a file
            with open("recalculation_times.txt", "a") as f:
                f.write(f"{num_switches},{avg_time}\n")  # Append the num_switches and avg_time
            return avg_time
        else:
            return 0
            
"""

    def save_all_data(self):
        self.save_congested_links_to_file('congested_links.pkl')
        self.save_utilization_data_to_file('utilization_data.pkl')
        self.save_path_changes_to_file('path_changes.pkl')
        self.save_flow_stats('flow_stats.pkl')
        self.save_metrics('paths.pkl', self.chosen_paths)
        
        
    @set_ev_cls(event.EventSwitchLeave, MAIN_DISPATCHER)
    def switch_leave_handler(self, ev):
        # This will reset variables when a switch leaves the network
        self.reset_variables()
        dpid = ev.switch.dp.id
        self.logger.info("Switch %s has left the network. Variables reset.", dpid)
    
    def save_metrics(self, filename, data):
        with open(filename, 'wb') as f:
            pickle.dump(data, f)  
             
    def save_congested_links_to_file(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.congested_links_over_time, f)

    def save_utilization_data_to_file(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.utilization_data), f)  # Convert defaultdict to dict

    def save_path_changes_to_file(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.path_changes), f)  # Convert defaultdict to dict
    
    def save_flow_stats(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.flow_stats), f)  # Convert defaultdict to dict
"""

