if __name__ == "__main__":
    import argparse
    
    import measurements.all.pw.values
    import measurements.land_sea_mask.data
    
    
    import util.logging
    logger = util.logging.logger

    ## configure arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--max_land_boxes', default=0, type=int)
    parser.add_argument('-d', '--debug', action='store_true', help='Print debug infos.')
    args = parser.parse_args()

    with util.logging.Logger(disp_stdout=args.debug):
        lsm = measurements.land_sea_mask.data.LandSeaMaskTMM()
        measurements.all.pw.values.points_near_water_mask(lsm, max_land_boxes=args.max_land_boxes)
