"""
Real-time tensile test tracking using ZED 2i stereo camera
"""
import pyzed.sl as sl
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s')

class TensileTracker:
    def __init__(self, mark_thickness=10, save_images=False, output_dir='./output'):
        """
        Initialize ZED camera and tracking parameters
        
        Parameters:
        -----------
        mark_thickness : int
            Approximate thickness of markers in pixels
        save_images : bool
            Whether to save captured frames
        output_dir : str
            Directory to save output data and images
        """
        self.mark_thickness = mark_thickness
        self.save_images = save_images
        self.output_dir = output_dir
        self.init_camera()
        self.strain_data = []
        self.reference_distance = None
        
    def init_camera(self):
        """Initialize and configure ZED camera"""
        self.zed = sl.Camera()
        
        # Create camera configuration
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD1080
        init_params.camera_fps = 30
        init_params.depth_mode = sl.DEPTH_MODE.ULTRA
        init_params.coordinate_units = sl.UNIT.MILLIMETER
        
        # Open the camera
        status = self.zed.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            logging.error(f"Camera initialization failed: {status}")
            exit(1)
            
        # Configure runtime parameters
        self.runtime_params = sl.RuntimeParameters()
        self.runtime_params.sensing_mode = sl.SENSING_MODE.STANDARD
        
        # Image containers
        self.image = sl.Mat()
        self.depth = sl.Mat()
        
        logging.info("ZED camera initialized successfully")
        
    def capture_frame(self):
        """Capture a new frame from ZED camera"""
        if self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
            self.zed.retrieve_measure(self.depth, sl.MEASURE.DEPTH)
            return True
        return False
    
    def get_intensity_profile(self, start_point, end_point, channel_idx=0):
        """
        Extract intensity profile along a line between two points
        
        Parameters:
        -----------
        start_point : tuple (x, y)
            Starting point coordinates
        end_point : tuple (x, y)
            Ending point coordinates
        channel_idx : int
            Color channel to use (0=R, 1=G, 2=B)
            
        Returns:
        --------
        profile : numpy array
            Intensity values along the line
        """
        # Get image as numpy array
        image_np = self.image.get_data()
        
        # Calculate number of points based on line length
        length = int(np.sqrt((end_point[0] - start_point[0])**2 + 
                            (end_point[1] - start_point[1])**2))
        
        # Generate line points
        x = np.linspace(start_point[0], end_point[0], length)
        y = np.linspace(start_point[1], end_point[1], length)
        
        # Extract intensity values using bilinear interpolation
        # Round to nearest pixel
        x_i = np.round(x).astype(int)
        y_i = np.round(y).astype(int)
        
        # Clip to image boundaries
        x_i = np.clip(x_i, 0, image_np.shape[1]-1)
        y_i = np.clip(y_i, 0, image_np.shape[0]-1)
        
        # Extract intensity values
        if len(image_np.shape) == 3:  # Color image
            profile = image_np[y_i, x_i, channel_idx]
        else:  # Grayscale image
            profile = image_np[y_i, x_i]
            
        # Create a profile function (similar to ConstFunction in original code)
        nodes = np.linspace(0, length, length)
        return nodes, profile
    
    def smooth_profile(self, nodes, values, window_size):
        """
        Apply smoothing to the intensity profile
        
        Parameters:
        -----------
        nodes : numpy array
            Position values
        values : numpy array
            Intensity values
        window_size : int
            Size of smoothing window
            
        Returns:
        --------
        smooth_func : callable
            Smoothed intensity function
        """
        def kernel(x, center, width):
            """Gaussian kernel"""
            return np.exp(-((x - center) / width)**2)
        
        def smooth_func(x):
            """Smoothed function at position x"""
            if np.isscalar(x):
                x = np.array([x])
                
            result = np.zeros_like(x, dtype=float)
            
            for i, xi in enumerate(x):
                weights = kernel(nodes, xi, window_size)
                if np.sum(weights) > 0:
                    result[i] = np.sum(weights * values) / np.sum(weights)
                else:
                    # Fallback to nearest neighbor
                    idx = np.argmin(np.abs(nodes - xi))
                    result[i] = values[idx]
                    
            return result[0] if len(result) == 1 else result
        
        return smooth_func
    
    def find_marker_positions(self, nodes, profile, n_markers=2):
        """
        Find dark markers on light background
        
        Parameters:
        -----------
        nodes : numpy array
            Position values
        profile : numpy array
            Intensity values
        n_markers : int
            Number of markers to find
            
        Returns:
        --------
        marker_positions : list
            Positions of detected markers
        """
        # Create smoothed profile function
        smooth_func = self.smooth_profile(nodes, profile, self.mark_thickness)
        
        # Split search space into sections based on number of markers
        segment_length = len(nodes) // n_markers
        marker_positions = []
        
        # Find minimum in each segment
        for i in range(n_markers):
            start_idx = i * segment_length
            end_idx = (i + 1) * segment_length if i < n_markers - 1 else len(nodes)
            
            # Find approximate minimum in segment
            segment = slice(start_idx, end_idx)
            min_idx = start_idx + np.argmin(profile[segment])
            x0 = nodes[min_idx]
            
            # Refine with optimization
            search_width = segment_length // 4
            left_bound = max(nodes[0], x0 - search_width)
            right_bound = min(nodes[-1], x0 + search_width)
            
            result = minimize_scalar(
                lambda x: float(smooth_func(x)),
                bounds=(left_bound, right_bound),
                method='bounded'
            )
            
            marker_positions.append(result.x)
            
        return marker_positions
    
    def calculate_strain(self, marker_positions):
        """
        Calculate engineering strain from marker positions
        
        Parameters:
        -----------
        marker_positions : list
            Positions of detected markers
            
        Returns:
        --------
        strain : float
            Current engineering strain
        """
        # Calculate current distance between markers
        current_distance = abs(marker_positions[1] - marker_positions[0])
        
        # Set reference distance if not yet established
        if self.reference_distance is None:
            self.reference_distance = current_distance
            return 0.0
        
        # Calculate engineering strain
        strain = (current_distance / self.reference_distance) - 1.0
        return strain
    
    def visualize_tracking(self, image, start_point, end_point, marker_positions, strain):
        """
        Visualize tracking results on image
        
        Parameters:
        -----------
        image : numpy array
            Image to draw on
        start_point, end_point : tuple
            Line endpoints
        marker_positions : list
            Detected marker positions
        strain : float
            Current strain value
            
        Returns:
        --------
        vis_image : numpy array
            Visualization image
        """
        # Make a copy to avoid modifying original
        vis_image = image.copy()
        
        # Draw measurement line
        cv2.line(vis_image, 
                (int(start_point[0]), int(start_point[1])),
                (int(end_point[0]), int(end_point[1])),
                (0, 255, 0), 2)
        
        # Calculate marker positions in image coordinates
        line_length = np.sqrt((end_point[0] - start_point[0])**2 + 
                             (end_point[1] - start_point[1])**2)
        
        for pos in marker_positions:
            # Convert 1D position to 2D coordinates
            t = pos / line_length
            x = int(start_point[0] + t * (end_point[0] - start_point[0]))
            y = int(start_point[1] + t * (end_point[1] - start_point[1]))
            
            # Draw marker position
            cv2.circle(vis_image, (x, y), 10, (0, 0, 255), -1)
            
        # Add strain text
        cv2.putText(vis_image, f"Strain: {strain:.4f}", 
                   (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                   (0, 0, 255), 2)
                   
        return vis_image
    
    def run_tracking(self, line_start, line_end, channel_idx=0):
        """
        Run real-time tensile test tracking
        
        Parameters:
        -----------
        line_start, line_end : tuple (x, y)
            Line endpoints for tracking
        channel_idx : int
            Color channel to use
            
        Returns:
        --------
        None
        """
        plt.figure(figsize=(12, 8))
        plt.ion()  # Interactive mode
        
        strain_history = []
        time_history = []
        start_time = time.time()
        
        try:
            while True:
                if not self.capture_frame():
                    logging.warning("Failed to capture frame")
                    continue
                    
                # Get image as numpy array
                image_np = self.image.get_data()
                
                # Get intensity profile
                nodes, profile = self.get_intensity_profile(
                    line_start, line_end, channel_idx)
                
                # Find marker positions
                marker_positions = self.find_marker_positions(nodes, profile)
                
                # Calculate strain
                strain = self.calculate_strain(marker_positions)
                self.strain_data.append(strain)
                
                # Visualization
                vis_image = self.visualize_tracking(
                    image_np, line_start, line_end, marker_positions, strain)
                
                # Update plot
                plt.clf()
                plt.subplot(2, 1, 1)
                plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
                plt.title("Tracking Visualization")
                
                # Plot strain over time
                current_time = time.time() - start_time
                time_history.append(current_time)
                strain_history.append(strain)
                
                plt.subplot(2, 1, 2)
                plt.plot(time_history, strain_history, 'b-')
                plt.xlabel("Time (seconds)")
                plt.ylabel("Engineering Strain")
                plt.title("Strain vs Time")
                plt.grid(True)
                
                plt.pause(0.01)
                
                # Save data periodically if needed
                if self.save_images and len(strain_history) % 30 == 0:
                    timestamp = int(time.time())
                    cv2.imwrite(f"{self.output_dir}/frame_{timestamp}.jpg", vis_image)
                    
                # Check for keyboard interrupt
                key = cv2.waitKey(1)
                if key == 27:  # ESC key
                    break
                    
        except KeyboardInterrupt:
            logging.info("Tracking interrupted by user")
        finally:
            # Close camera and save results
            self.zed.close()
            self.save_results(time_history, strain_history)
            plt.ioff()
            
    def save_results(self, time_data, strain_data):
        """Save tracking results to file"""
        results = np.column_stack((time_data, strain_data))
        np.savetxt(f"{self.output_dir}/strain_results.csv", 
                  results, delimiter=",", 
                  header="time_seconds,engineering_strain")
        
        # Create final plot
        plt.figure(figsize=(10, 6))
        plt.plot(time_data, strain_data, 'b-', linewidth=2)
        plt.xlabel("Time (seconds)")
        plt.ylabel("Engineering Strain")
        plt.title("Tensile Test Results")
        plt.grid(True)
        plt.savefig(f"{self.output_dir}/strain_plot.png", dpi=300)
        
        logging.info(f"Results saved to {self.output_dir}")


def main():
    """Main function to run the tracker"""
    import os
    import argparse
    
    parser = argparse.ArgumentParser(description="ZED Camera Tensile Test Tracker")
    parser.add_argument("--mark-thickness", type=int, default=10,
                        help="Approximate thickness of markers in pixels")
    parser.add_argument("--save-images", action="store_true",
                        help="Save captured frames")
    parser.add_argument("--output-dir", type=str, default="./output",
                        help="Directory to save output data")
    parser.add_argument("--line", type=int, nargs=4, 
                        default=[640, 300, 640, 700],
                        help="Line coordinates (x1,y1,x2,y2)")
    args = parser.parse_args()
    
    # Create output directory if needed
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Initialize tracker
    tracker = TensileTracker(
        mark_thickness=args.mark_thickness,
        save_images=args.save_images,
        output_dir=args.output_dir
    )
    
    # Run tracking
    line_start = (args.line[0], args.line[1])
    line_end = (args.line[2], args.line[3])
    tracker.run_tracking(line_start, line_end)

if __name__ == "__main__":
    main()
