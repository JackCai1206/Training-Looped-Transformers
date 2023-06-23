cp -rf /home/groups/cafa-5-group/Training-Looped-Transformers /staging/groups/cafa-5-group/

truncate -s 0 loop_trans.err
truncate -s 0 loop_trans.out
truncate -s 0 loop_trans.log

condor_submit train.sub
