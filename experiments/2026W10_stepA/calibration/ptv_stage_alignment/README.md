# PTV–LDV coordinate alignment calibration

Maps PTV microscope image coordinates to LDV physical channel-length coordinates.

## Applicable datasets

- **LDV**: test10 voltage sweep, Mar 7 2026
  `$LDV_DATA_ROOT/20260307experimentB/test10_1907_*Vpp*.tdms`
- **PTV**: 260317piv, Mar 17 2026
  `D:/OneDrive - Lund University/Data/260317piv/{5,10,15}Vpp/`
  Processed output: `C:/Users/tatsu/Documents/particle-tracking/output/{5,10,15}Vpp/`
- Same chip, moved from LDV bench to microscope between sessions

## Method

The PZT edges are visible as shadows in the PTV camera and their physical
positions are known from the LDV scanner coordinates. Two reference points
give the coordinate mapping.

## Reference points

| Edge | Physical x (mm) | PTV camera pixel | PTV stage x (mm) |
|------|-----------------|-----------------|-------------------|
| Inlet (upstream) | 5.6 | 1289 | 2.409 |
| Outlet (downstream) | 11.6 | 1070 | -3.494 |

PTV data capture stage position: x = -0.7699 mm.

## Coordinate mapping

```
physical_x_mm = -stage_x_mm - pixel * 0.00065 + offset
```

where pixel_size = 0.65 um/px.

Offset from inlet: 8.847 mm
Offset from outlet: 8.802 mm
Average offset: **8.825 mm** (0.045 mm residual = calibration precision).

For PTV data (stage = -0.7699 mm, X in um from particle centroid):
```
physical_x_mm = 9.595 - X_um / 1000
```

PTV X range 55-1853 um -> physical x = 7.74-9.54 mm.

## Sign convention

- Increasing PTV stage x -> toward inlet (upstream)
- Increasing pixel number -> toward inlet
- physical_x increases toward outlet (downstream)
- Hence both stage and pixel enter with negative sign

## Files

| File | Description |
|------|-------------|
| `ptv_camera_inlet_edge_stage_x2.409mm.png` | PTV camera view with PZT inlet edge shadow at pixel 1289 |
| `ptv_camera_outlet_edge_stage_x-3.494mm.nd2` | PTV camera view with PZT outlet edge shadow at pixel 1070 |
| `ptv_data_capture_stage_x-0.7699mm.nd2` | PTV data capture session metadata (stage position) |
| `ldv_laser_at_inlet_pzt_edge_y5.6mm.jpg` | LDV camera view with laser at inlet PZT edge (y=5.6 mm) |
| `ldv_laser_at_outlet_pzt_edge_y11.6mm.jpg` | LDV camera view with laser at outlet PZT edge (y=11.6 mm) |

## Source locations

- PTV camera images: `particle-tracking/output/location_tiff/` and `260317piv/location/`
- LDV camera images: `260318_ldv/`
- PTV data: `260317piv/Captured 2.nd2`
