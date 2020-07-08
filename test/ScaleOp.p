#Parameter file to run a simple scale operator for testing
RUN: Add scale=1e-4 < input.H > output.H
RUN: echo tmp_random_name0