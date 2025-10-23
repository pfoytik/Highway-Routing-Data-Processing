#!/usr/bin/env python3
"""
Highway Routing Data Analysis
Processes highway vehicle routing data to calculate travel times and map to road segments.
"""

import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import math
from geopy.distance import geodesic

class HighwayRoutingAnalyzer:
    """
    Main class for processing highway routing data and mapping to road segments.
    """
    
    def __init__(self, links_xml_path: str, segments_xml_path: str, csv_path: str):
        self.links_xml_path = links_xml_path
        self.segments_xml_path = segments_xml_path
        self.csv_path = csv_path
        self.links_data = {}
        self.segments_data = {}
        self.routing_data = None
        
    def parse_links_xml(self) -> Dict:
        """
        Parse links.xml file to extract link information with coordinates.
        Returns dictionary with link_id as key and link data as value.
        """
        print("Parsing links.xml file...")
        tree = ET.parse(self.links_xml_path)
        root = tree.getroot()
        
        links = {}
        for link in root.findall('Link'):
            link_id = link.find('ID').text
            a_node = link.find('ANode').text
            b_node = link.find('BNode').text
            length = float(link.find('Length').text)
            name = link.find('Name').text if link.find('Name') is not None else ""
            
            links[link_id] = {
                'id': link_id,
                'a_node': a_node,
                'b_node': b_node,
                'length': length,
                'name': name
            }
        
        print(f"Parsed {len(links)} links")
        return links
    
    def parse_segments_xml(self) -> Dict:
        """
        Parse segments.xml file to extract segment information.
        Returns dictionary with segment_id as key and segment data as value.
        """
        print("Parsing segments.xml file...")
        tree = ET.parse(self.segments_xml_path)
        root = tree.getroot()
        
        segments = {}
        for segment in root.findall('Segment'):
            segment_id = segment.find('ID').text
            link_id = segment.find('Link_ID').text
            length = float(segment.find('Length').text)
            position = float(segment.find('Position').text)
            speed_limit = float(segment.find('Speed_Limit').text) if segment.find('Speed_Limit') is not None else 0
            
            segments[segment_id] = {
                'id': segment_id,
                'link_id': link_id,
                'length': length,
                'position': position,
                'speed_limit': speed_limit
            }
        
        print(f"Parsed {len(segments)} segments")
        return segments
    
    def load_routing_data(self) -> pd.DataFrame:
        """
        Load and process the CSV routing data.
        Calculate travel times and prepare data for segment mapping.
        """
        print("Loading routing data...")
        df = pd.read_csv(self.csv_path)
        
        # Convert timestamps to datetime
        df['Entrance_TIMESTAMP'] = pd.to_datetime(df['Entrance_TIMESTAMP'])
        df['Exit_TIMESTAMP'] = pd.to_datetime(df['Exit_TIMESTAMP'])
        
        # Calculate travel time in seconds
        df['Travel_Time_Seconds'] = (df['Exit_TIMESTAMP'] - df['Entrance_TIMESTAMP']).dt.total_seconds()
        
        # Calculate distance using geodesic distance
        df['Distance_Meters'] = df.apply(
            lambda row: geodesic(
                (row['Entrance_LAT'], row['Entrance_LON']),
                (row['Exit_LAT'], row['Exit_LON'])
            ).meters, axis=1
        )
        
        # Calculate average speed (m/s)
        df['Average_Speed_mps'] = df['Distance_Meters'] / df['Travel_Time_Seconds']
        
        print(f"Loaded {len(df)} routing records")
        return df
    
    def calculate_distance_to_segment(self, lat: float, lon: float, 
                                    segment_start_lat: float, segment_start_lon: float,
                                    segment_end_lat: float, segment_end_lon: float) -> float:
        """
        Calculate the minimum distance from a point to a line segment.
        Uses the perpendicular distance to the line segment.
        """
        # Convert to radians for calculation
        def to_rad(deg):
            return deg * math.pi / 180
        
        # Point to line segment distance calculation
        A = (segment_start_lat, segment_start_lon)
        B = (segment_end_lat, segment_end_lon)
        P = (lat, lon)
        
        # Use geodesic distance for accuracy
        dist_AP = geodesic(A, P).meters
        dist_BP = geodesic(B, P).meters
        dist_AB = geodesic(A, B).meters
        
        # If the segment has no length, return distance to the point
        if dist_AB == 0:
            return dist_AP
        
        # Calculate perpendicular distance using the cross product method
        # This is an approximation for small distances
        s = (dist_AP**2 - dist_BP**2 + dist_AB**2) / (2 * dist_AB)
        
        if s <= 0:
            return dist_AP
        elif s >= dist_AB:
            return dist_BP
        else:
            # Use Heron's formula to find the height of the triangle
            semi_perimeter = (dist_AP + dist_BP + dist_AB) / 2
            area = math.sqrt(semi_perimeter * (semi_perimeter - dist_AP) * 
                           (semi_perimeter - dist_BP) * (semi_perimeter - dist_AB))
            return 2 * area / dist_AB
    
    def find_closest_segment(self, lat: float, lon: float, 
                           max_distance: float = 1000) -> Optional[str]:
        """
        Find the closest road segment to a given lat/lon coordinate.
        Returns segment_id if found within max_distance meters, None otherwise.
        """
        min_distance = float('inf')
        closest_segment_id = None
        
        for segment_id, segment in self.segments_data.items():
            link_id = segment['link_id']
            if link_id in self.links_data:
                link = self.links_data[link_id]
                
                # For this example, we'll assume we have node coordinates
                # In a real implementation, you'd need to get these from a nodes.xml file
                # or calculate segment positions along the link
                
                # Placeholder: using link start/end as segment bounds
                # This would need to be refined based on segment position within link
                segment_start_lat = float(link.get('start_lat', 0))  # Would need actual node coordinates
                segment_start_lon = float(link.get('start_lon', 0))
                segment_end_lat = float(link.get('end_lat', 0))
                segment_end_lon = float(link.get('end_lon', 0))
                
                if segment_start_lat == 0 and segment_start_lon == 0:
                    # Skip if no coordinates available
                    continue
                
                distance = self.calculate_distance_to_segment(
                    lat, lon, segment_start_lat, segment_start_lon,
                    segment_end_lat, segment_end_lon
                )
                
                if distance < min_distance and distance <= max_distance:
                    min_distance = distance
                    closest_segment_id = segment_id
        
        return closest_segment_id
    
    def map_routes_to_segments(self) -> pd.DataFrame:
        """
        Map each route in the routing data to the closest road segments.
        """
        print("Mapping routes to road segments...")
        
        # Add columns for segment mapping
        self.routing_data['Entrance_Segment_ID'] = None
        self.routing_data['Exit_Segment_ID'] = None
        self.routing_data['Entrance_Distance_to_Segment'] = None
        self.routing_data['Exit_Distance_to_Segment'] = None
        
        for idx, row in self.routing_data.iterrows():
            # Map entrance location
            entrance_segment = self.find_closest_segment(
                row['Entrance_LAT'], row['Entrance_LON']
            )
            
            # Map exit location
            exit_segment = self.find_closest_segment(
                row['Exit_LAT'], row['Exit_LON']
            )
            
            self.routing_data.at[idx, 'Entrance_Segment_ID'] = entrance_segment
            self.routing_data.at[idx, 'Exit_Segment_ID'] = exit_segment
            
            if idx % 100 == 0:
                print(f"Processed {idx} records...")
        
        print("Completed segment mapping")
        return self.routing_data
    
    def process_data(self) -> pd.DataFrame:
        """
        Main processing pipeline.
        """
        print("Starting highway routing data analysis...")
        
        # Load and parse data
        self.links_data = self.parse_links_xml()
        self.segments_data = self.parse_segments_xml()
        self.routing_data = self.load_routing_data()
        
        # Map routes to segments
        result_data = self.map_routes_to_segments()
        
        return result_data
    
    def save_results(self, output_path: str):
        """
        Save the processed results to a CSV file.
        """
        print(f"Saving results to {output_path}")
        self.routing_data.to_csv(output_path, index=False)
        print("Results saved successfully")


def main():
    """
    Main function to run the highway routing analysis.
    """
    # File paths
    links_xml_path = "fleet_kincadeFire/834602066/links.xml"
    segments_xml_path = "fleet_kincadeFire/834602066/segments.xml"
    csv_path = "PRJ-3226/Highway_Vehicle_Routing_Data_R1.csv"
    output_path = "highway_routing_with_segments.csv"
    
    # Create analyzer instance
    analyzer = HighwayRoutingAnalyzer(links_xml_path, segments_xml_path, csv_path)
    
    try:
        # Process the data
        result_data = analyzer.process_data()
        
        # Save results
        analyzer.save_results(output_path)
        
        # Print summary statistics
        print("\n=== Analysis Summary ===")
        print(f"Total records processed: {len(result_data)}")
        print(f"Records with entrance segment mapped: {result_data['Entrance_Segment_ID'].notna().sum()}")
        print(f"Records with exit segment mapped: {result_data['Exit_Segment_ID'].notna().sum()}")
        print(f"Records with both segments mapped: {(result_data['Entrance_Segment_ID'].notna() & result_data['Exit_Segment_ID'].notna()).sum()}")
        print(f"Average travel time: {result_data['Travel_Time_Seconds'].mean():.2f} seconds")
        print(f"Average distance: {result_data['Distance_Meters'].mean():.2f} meters")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        raise


if __name__ == "__main__":
    main()