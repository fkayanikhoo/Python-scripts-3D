parallel -j32 --delay 5 --line-buffer \
      python slice_with_vectors_fast.py {}  \
      --scalar log_Ehat_s log_rho_s log_uint_s \
      --slice XZ XY \
      -o alpha22 \
      ::: data/simext{0401..0570..1}.dat