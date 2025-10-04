import torch
import numpy as np

from helpers.PointModel import PointModel

def green_spill_in_SH(model: PointModel, threshold: float = 0.2, 
                      view_dirs: np.ndarray = None, num_sample_views: int = 8) -> np.ndarray:
    """
    Detect points with green-dominant SH coefficients (view-dependent spill).
    
    Args:
        model: PointModel with SH coefficients
        threshold: Relative threshold for green dominance
        view_dirs: Optional specific viewing directions to check (N_views, 3)
        num_sample_views: Number of random viewing directions to sample if view_dirs not provided
    
    Returns:
        Boolean array indicating green spill points
    """
    
    if model._features_rest.numel() == 0:
        return np.zeros(model._features_dc.shape[0], dtype=bool)
    
    with torch.no_grad():
        # Get DC and SH coefficients
        dc = model._features_dc.cpu().numpy()  # (N, 3) - base color
        rest = model._features_rest.cpu().numpy()  # (N, SH_coeffs, 3)
        
        N = dc.shape[0]
        
        # Method 1: Analyze DC component for obvious green bias
        dc_green_mask = detect_dc_green_bias(dc, threshold)
        
        # Method 2: Sample multiple viewing directions to detect view-dependent green spill
        if view_dirs is None:
            # Generate sample viewing directions (roughly uniform on sphere)
            view_dirs = generate_sample_views(num_sample_views)
        
        view_dependent_mask = detect_view_dependent_green_spill(
            dc, rest, view_dirs, threshold
        )
        
        # Method 3: Check for anomalous SH coefficient patterns
        sh_pattern_mask = detect_anomalous_sh_patterns(rest, threshold)
        
        # Combine all detection methods
        combined_mask = dc_green_mask | view_dependent_mask | sh_pattern_mask
        
        return combined_mask

def detect_dc_green_bias(dc: np.ndarray, threshold: float) -> np.ndarray:
    """Detect obvious green bias in DC (base) color component."""
    # Handle different DC shapes
    if dc.shape[1] == 1:
        # Single channel - can't detect green spill
        return np.zeros(dc.shape[0], dtype=bool)
    elif dc.shape[1] == 3:
        # RGB channels
        r, g, b = dc[:, 0], dc[:, 1], dc[:, 2]
    else:
        # Unexpected format - skip DC analysis
        return np.zeros(dc.shape[0], dtype=bool)
    
    # Method 1: Ratio-based for brighter colors
    brightness = r + g + b
    bright_mask = brightness > 0.1  # Only use ratios for reasonably bright pixels
    
    ratio_green_dominant = np.zeros_like(bright_mask)
    ratio_green_dominant[bright_mask] = (
        (g[bright_mask] > r[bright_mask] * (1 + threshold)) & 
        (g[bright_mask] > b[bright_mask] * (1 + threshold))
    )
    
    # Method 2: Absolute difference for darker colors
    dark_mask = brightness <= 0.1
    min_green_diff = threshold * 0.05  # Minimum absolute green difference
    
    abs_green_dominant = np.zeros_like(dark_mask)
    if np.any(dark_mask):
        g_diff_r = g[dark_mask] - r[dark_mask]
        g_diff_b = g[dark_mask] - b[dark_mask]
        abs_green_dominant[dark_mask] = (
            (g_diff_r > min_green_diff) & (g_diff_b > min_green_diff)
        )
    
    # Method 3: Statistical outlier detection for very subtle spill
    # Convert to LAB color space approximation for better perceptual analysis
    lab_green_bias = detect_perceptual_green_bias(r, g, b, threshold)
    
    # Method 4: Check for green shift in darker regions
    # Dark green spill often shows as green being the dominant channel even in low light
    dark_green_dominant = dark_mask & (g > r) & (g > b) & (g > threshold * 0.02)
    
    return ratio_green_dominant | abs_green_dominant | lab_green_bias | dark_green_dominant

def detect_perceptual_green_bias(r: np.ndarray, g: np.ndarray, b: np.ndarray, threshold: float) -> np.ndarray:
    """Detect green bias using perceptual color differences."""
    # Simple RGB to approximate LAB conversion for better perceptual analysis
    # This helps detect green tints that might not be obvious in RGB space
    
    # Normalize to prevent issues with very dark colors
    rgb_sum = r + g + b + 1e-8
    r_norm = r / rgb_sum
    g_norm = g / rgb_sum  
    b_norm = b / rgb_sum
    
    # Expected neutral would be roughly equal proportions
    expected_prop = 1/3
    
    # Check if green proportion is significantly higher than expected
    green_excess = g_norm - expected_prop
    other_deficit = (expected_prop - r_norm) + (expected_prop - b_norm)
    
    # Green bias detected if green is significantly over-represented
    perceptual_bias = (green_excess > threshold * 0.1) & (other_deficit > threshold * 0.05)
    
    return perceptual_bias

def detect_view_dependent_green_spill(dc: np.ndarray, rest: np.ndarray, 
                                     view_dirs: np.ndarray, threshold: float) -> np.ndarray:
    """Detect green spill that appears in specific viewing directions."""
    N = dc.shape[0]
    spill_detected = np.zeros(N, dtype=bool)
    
    # Skip if DC doesn't have RGB channels
    if dc.shape[1] != 3:
        return spill_detected
    
    for view_dir in view_dirs:
        # Evaluate SH for this viewing direction
        rendered_color = evaluate_sh_at_direction(dc, rest, view_dir)
        
        # Check for green spill in this view
        r, g, b = rendered_color[:, 0], rendered_color[:, 1], rendered_color[:, 2]
        
        # Detect green spill in this specific view
        view_green_spill = (g > r * (1 + threshold)) & (g > b * (1 + threshold))
        
        # Also check for color shift towards green compared to base color
        base_r, base_g, base_b = dc[:, 0], dc[:, 1], dc[:, 2]
        green_shift = (g - base_g) > (np.maximum(r - base_r, b - base_b) + threshold)
        
        spill_detected |= (view_green_spill | green_shift)
    
    return spill_detected

def detect_anomalous_sh_patterns(rest: np.ndarray, threshold: float) -> np.ndarray:
    """Detect anomalous patterns in SH coefficients that suggest green spill."""
    if rest.shape[1] == 0:
        return np.zeros(rest.shape[0], dtype=bool)
    
    # Handle different rest shapes
    if rest.ndim != 3:
        return np.zeros(rest.shape[0], dtype=bool)
    
    # Check if we have RGB channels
    if rest.shape[2] < 3:
        return np.zeros(rest.shape[0], dtype=bool)
    
    # Analyze the energy distribution across SH bands
    # Green spill often shows up as unusual energy in higher-order SH coefficients
    
    # Calculate energy per color channel across all SH coefficients
    energy_per_channel = (rest ** 2).sum(axis=1)  # (N, 3)
    r_energy, g_energy, b_energy = energy_per_channel[:, 0], energy_per_channel[:, 1], energy_per_channel[:, 2]
    
    # Check for disproportionate green energy
    total_energy = energy_per_channel.sum(axis=1)
    green_ratio = g_energy / (total_energy + 1e-8)
    
    # Detect points where green takes up too much of the total SH energy
    anomalous_green_energy = green_ratio > (1/3 + threshold)
    
    # Check for inconsistent signs in green SH coefficients (can indicate spill)
    if rest.shape[1] > 1:
        green_coeffs = rest[:, :, 1]  # All green SH coefficients
        sign_changes = np.sum(np.diff(np.sign(green_coeffs), axis=1) != 0, axis=1)
        # Too many sign changes might indicate problematic coefficients
        inconsistent_signs = sign_changes > (rest.shape[1] * 0.6)
    else:
        inconsistent_signs = np.zeros(rest.shape[0], dtype=bool)
    
    return anomalous_green_energy | inconsistent_signs

def generate_sample_views(num_views: int) -> np.ndarray:
    """Generate roughly uniform sample directions on unit sphere."""
    # Use Fibonacci sphere for good uniform distribution
    indices = np.arange(0, num_views, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / num_views)  # Inclination
    theta = np.pi * (1 + 5**0.5) * indices  # Azimuth (golden angle)
    
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    
    return np.column_stack([x, y, z])

def evaluate_sh_at_direction(dc: np.ndarray, rest: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Evaluate spherical harmonics at a specific viewing direction."""
    # This is a simplified version - you'll need to implement proper SH evaluation
    # based on your specific SH basis functions and degree
    
    # For now, approximating with DC + first-order directional component
    result = dc.copy()
    
    if rest.shape[1] > 0:
        # Add contribution from first SH band (assuming it represents directional variation)
        # This is simplified - proper implementation needs full SH basis evaluation
        dir_contrib = rest[:, 0, :] * direction[0]  # Simplified directional term
        if rest.shape[1] > 1:
            dir_contrib += rest[:, 1, :] * direction[1]
        if rest.shape[1] > 2:
            dir_contrib += rest[:, 2, :] * direction[2]
        
        result += dir_contrib
    
    # Clamp to reasonable color range
    result = np.clip(result, 0, 1)
    
    return result


# Enhanced version with better dark green detection
def green_spill_statistical_enhanced(model: PointModel, threshold: float = 1.5) -> np.ndarray:
    """
    Enhanced statistical detection with better handling of dark green spill.
    """
    if model._features_rest.numel() == 0:
        return np.zeros(model._features_dc.shape[0], dtype=bool)
    
    with torch.no_grad():
        dc = model._features_dc.cpu().numpy()
        rest = model._features_rest.cpu().numpy()
        
        print(f"DC shape: {dc.shape}, Rest shape: {rest.shape}")
        
        if dc.shape[1] == 1 or dc.shape[1] != 3:
            return np.zeros(dc.shape[0], dtype=bool)
            
        r, g, b = dc[:, 0], dc[:, 1], dc[:, 2]
        
        # Separate analysis for different brightness levels
        brightness = r + g + b
        
        # Method 1: Ratio-based for bright pixels
        bright_mask = brightness > np.percentile(brightness, 25)  # Top 75% brightness
        ratio_spill = np.zeros(len(r), dtype=bool)
        
        if np.any(bright_mask):
            gr_ratio = g[bright_mask] / (r[bright_mask] + 1e-8)
            gb_ratio = g[bright_mask] / (b[bright_mask] + 1e-8)
            
            gr_mean, gr_std = np.mean(gr_ratio), np.std(gr_ratio)
            gb_mean, gb_std = np.mean(gb_ratio), np.std(gb_ratio)
            
            gr_outliers = gr_ratio > (gr_mean + threshold * gr_std)
            gb_outliers = gb_ratio > (gb_mean + threshold * gb_std)
            
            ratio_spill[bright_mask] = gr_outliers | gb_outliers
        
        # Method 2: Absolute difference for dark pixels
        dark_mask = brightness <= np.percentile(brightness, 25)  # Bottom 25% brightness
        abs_spill = np.zeros(len(r), dtype=bool)
        
        if np.any(dark_mask):
            # For dark pixels, look for absolute green excess
            g_excess_r = g[dark_mask] - r[dark_mask]
            g_excess_b = g[dark_mask] - b[dark_mask]
            
            # Statistical analysis of green excess in dark regions
            if len(g_excess_r) > 1:
                gr_excess_mean, gr_excess_std = np.mean(g_excess_r), np.std(g_excess_r)
                gb_excess_mean, gb_excess_std = np.mean(g_excess_b), np.std(g_excess_b)
                
                # Lower threshold for dark regions
                dark_threshold = threshold * 0.7
                
                gr_excess_outliers = g_excess_r > (gr_excess_mean + dark_threshold * gr_excess_std)
                gb_excess_outliers = g_excess_b > (gb_excess_mean + dark_threshold * gb_excess_std)
                
                abs_spill[dark_mask] = gr_excess_outliers & gb_excess_outliers
        
        # Method 3: Normalized green proportion analysis
        total_color = r + g + b + 1e-8
        green_proportion = g / total_color
        
        # Statistical analysis of green proportions
        gp_mean, gp_std = np.mean(green_proportion), np.std(green_proportion)
        prop_outliers = green_proportion > (gp_mean + threshold * gp_std)
        
        # Also check if green proportion is unnaturally high (>40% in any pixel)
        high_green_prop = green_proportion > 0.4
        
        # Method 4: SH coefficient analysis if available
        sh_spill = np.zeros(len(r), dtype=bool)
        if rest.shape[1] > 0 and rest.ndim == 3 and rest.shape[2] >= 3:
            rest_energy = (rest ** 2).sum(axis=1)
            r_energy, g_energy, b_energy = rest_energy[:, 0], rest_energy[:, 1], rest_energy[:, 2]
            
            # Analyze green energy patterns
            total_sh_energy = r_energy + g_energy + b_energy + 1e-8
            green_sh_prop = g_energy / total_sh_energy
            
            gsh_mean, gsh_std = np.mean(green_sh_prop), np.std(green_sh_prop)
            sh_spill = green_sh_prop > (gsh_mean + threshold * gsh_std)
        
        # Combine all methods with weighted voting
        combined_mask = (
            ratio_spill |  # Good for bright regions
            abs_spill |    # Good for dark regions  
            (prop_outliers & (brightness < np.percentile(brightness, 50))) |  # Proportional analysis for mid-dark
            high_green_prop |  # Catch extreme cases
            sh_spill       # SH pattern analysis
        )
        
        return combined_mask