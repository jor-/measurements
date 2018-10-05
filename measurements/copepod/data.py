import csv

import util.logging


def load(file, pgc):

    # for parsing cvs lines
    dialect = csv.unix_dialect
    dialect.skipinitialspace = True

    def read_csv_line(line):
        return next(csv.reader((line,), dialect=dialect))

    # read all lines
    util.logging.debug('Reading all values from {}.'.format(file))

    with open(file, mode='r') as csv_file:
        lines = csv_file.readlines()

    # read metadata
    util.logging.debug('Got {} lines from {}.'.format(len(lines), file))

    assert lines[0] == '#\n'
    assert lines[1] == '#  This data file is a COPEPOD planktgon group compilation.  ( Information on format and content are available at:  http://www.st.nmfs.noaa.gov/plankton/atlas )\n'
    assert lines[2] == '#\n'
    assert lines[3] == '#SHP-CRUISE,YEAR,MON,DAY,TIMEgmt,TIMEloc,LATITUDE,LONGITDE,UPPER_Z,LOWER_Z,T,GEAR,MESH,NMFS_PGC,ITIS_TSN,MOD,LIF,PSC,SEX,V,Water Strained,Original-VALUE,Orig-UNITS,VALUE-per-volu,UNITS,F1,F2,F3,F4,VALUE-per-area,UNITS,F1,F2,F3,F4,SCIENTIFIC NAME -[ modifiers ]-,RECORD-ID,DATASET-ID,SHIP,PROJ,INST,\n'
    assert lines[4] == '#----------,----,---,---,-------,-------,--------,--------,-------,-------,-,----,----,--------,--------,---,---,---,---,-,--------------,--------------,----------,--------------,-----,--,--,--,--,--------------,-----,--,--,--,--,-------------------------------,---------,----------,----,----,----,\n'

    metadata = read_csv_line(lines[3])
    PGC_VARIABLE_NAME = 'NMFS_PGC'
    pgc_variable_index = metadata.index(PGC_VARIABLE_NAME)

#    # filter PGC
#    lines = lines[5:]
#    lines = tuple(read_csv_line(line) for line in lines)
#    lines = tuple(line for line in lines if int(line[pgc_index]) == pgc)
#    util.logging.debug('Got {} lines after filtering only pgc {}.'.format(len(lines), pgc))

    # filter flags
    VALID_FLAG = 0
    TOO_FEW_MEASUREMENTS_FOR_QUALITY_CHECK_FLAG = 1
    ZERO_VALUE_FLAG = -1
    OKAY_FLAGS = (VALID_FLAG, TOO_FEW_MEASUREMENTS_FOR_QUALITY_CHECK_FLAG, ZERO_VALUE_FLAG)

    def all_indices(metadata, value):
        return tuple(i for i, x in enumerate(metadata) if x == value)

    FLAG_VARIABLE_NAMES = ('F1', 'F2', 'F3', 'F4')
    flag_variable_indices = (index for flag_variable_name in FLAG_VARIABLE_NAMES for index in all_indices(metadata, flag_variable_name))
#    for flag_variable_index in flag_variable_indices:
#        lines = tuple(line for line in lines if int(line[flag_variable_index]) in OKAY_FLAGS)
#    util.logging.debug('Got {} lines after filtering only valid flags {}.'.format(len(lines), OKAY_FLAGS))

#    # convert values
    value_index = metadata.index('VALUE-per-volu')
    unit_index = metadata.index('UNITS')
#    for line in lines:
#        value = line[value_index]
#        unit = line[unit_index]
#        print(value)
#        print(unit)
#        assert unit == '#/ml'

    lines_index_offset = 5
    for i, line in enumerate(lines[lines_index_offset:]):
        # convert line to value tuple
        line = read_csv_line(line)
        line_index = i + lines_index_offset

        try:
            # check pgc
            pgc_value = int(line[pgc_variable_index])
            if pgc_value != pgc:
                raise WrongValueError(line_index, metadata[pgc_variable_index], pgc_value, pgc)

            # check flags
            for flag_variable_index in flag_variable_indices:
                flag_value = int(line[flag_variable_index])
                if flag_value not in OKAY_FLAGS:
                    raise WrongValueError(line_index, metadata[flag_variable_index], flag_value, OKAY_FLAGS)

            # check value and unit
            value = line[value_index]
            try:
                value = float(value)
            except ValueError:
                raise InvalidValueError(line_index, metadata[value_index], value)

            unit = line[unit_index]
            assert unit == '#/ml'
        except WrongValueError as e:
            util.logging.debug(e)
        except InvalidValueError as e:
            util.logging.warning(e)



#        # check pgc
#        pgc_line = int(line[pgc_index])
#        if pgc_line != pgc:
#            util.logging.debug('Line {line_index} PGC {PGC_line} does not match desired PGC {PGC_desired}.'.format(line_index=line_index, pgc_line=pgc_line,PGC_desired=pgc))
#            break
#
#        for flag_index in flag_indices:
#            lines = tuple(line for line in lines if int(line[flag_index]) in OKAY_FLAGS)





class InvalidValueError(ValueError):
    def __init__(self, line_index, name, value):
        message = 'The value "{value}" of variable "{name}" in line {line_index} is invalid.'.format(value=value, line_index=line_index, name=name)
        self.message = message
        self.line_index = line_index
        self.value = value
        super().__init__(message)



class WrongValueError(ValueError):
    def __init__(self, line_index, name, value, desired_value):
        message = 'The value "{value}" of variable "{name}" in line {line_index} does not match the desired value {desired_value}.'.format(value=value, line_index=line_index, name=name, desired_value=desired_value)
        self.message = message
        self.line_index = line_index
        self.value = value
        self.desired_value = desired_value
        super().__init__(message)