/************************************************************


In the following code below, you have the data preprocessing
for the model. These steps follows a very detailed process,
each section of code should be run one step at a time. If not,
it may result in an error in the model as the data tables that
are being created may not be release from the CAS Server.


 ***************************************************************/
/******************  Create Libref Statement  ****************/
libname ps '/home/student/Courses/PROC_SUPERLEARNER/data';
libname out '/home/student/Courses/PROC_SUPERLEARNER/output';

/******  Start CAS Session   *******/
cas mySession sessopts=(caslib=casuser);
libname mycas cas caslib="casuser";

/***    LOAD DATA IN-MEMORY     *******/
data ps.tele_comms_clean;
    set ps.commsdata;
run;

/****** SUMMARY STATISTIC   ***********/
;

proc means data=ps.tele_comms_clean N Mean std min max skewness kurtosis
    descending ;
run;

/************ DATA ENGINEERING / FEATURE EXTRACTION ******/
;

/**  Handling the negative values **/
;

data out.tele_comms_clean;
    set ps.commsdata;

    /**  The original data with negative values   **/
    ;

    array old_limits(*) tot_mb_data_roam_curr seconds_of_data_norm
        lifetime_value bill_data_usg_m03 bill_data_usg_m06
        voice_tot_bill_mou_curr mb_data_usg_roamm01 mb_data_usg_roamm02
        mb_data_usg_roamm03 mb_data_usg_m01 mb_data_usg_m02 mb_data_usg_m03
        calls_total calls_in_pk calls_out_pk calls_in_offpk calls_out_offpk
        mb_data_ndist_mo6m data_device_age mou_onnet_pct_MOM mou_total_pct_MOM;

    array new_limits(*) tot_mb_data_roam_curr seconds_of_data_norm
        lifetime_value bill_data_usg_m03 bill_data_usg_m06
        voice_tot_bill_mou_curr mb_data_usg_roamm01 mb_data_usg_roamm02
        mb_data_usg_roamm03 mb_data_usg_m01 mb_data_usg_m02 mb_data_usg_m03
        calls_total calls_in_pk calls_out_pk calls_in_offpk calls_out_offpk
        mb_data_ndist_mo6m data_device_age mou_onnet_pct_MOM mou_total_pct_MOM;

    /**  Apply rule to set negative values to zero  **/
    ;

    do i=1 to dim(old_limits);

        /**  If the old value is a negative value then we will set the new value to 0  **/
        if old_limits[i] < 0 then new_limits[i]=0;

        /** If the value is greater than 0, then we keep the original value **/
        else old_limits[i]=new_limits[i];

    end;
    /* Drop variables (outside loops) */
    drop i upsell_xsell city verbatims city_lat city_long data_usage_amt
        mou_onnet_6m_normal mou_roam_6m_normal region_lat region_long state_lat
        state_long tweedie_adjusted zip_lat zip_long zipcode_primary;
run;

/****  Change the Churn from (0 & 1) to (churn & stay)   ****/

data out.tele_comms_clean;
    set out.tele_comms_clean;
    length CustChurn $8;
    if churn = 1 then CustChurn='Churn';
    if churn = 0 then CustChurn='Stay';
    drop churn;
    rename CustChurn=churn;
run;

/***********  STOP!!!!!!    STOP!!!!!!! ONLY FOR PART 1  *****/
/*******  DATA PARTITION: SPLITTING OUR DATA INTO A 70/30 SPLIT  *****/
/********************************************************************/
;

proc partition data=ps.tele_comms_clean samppct=60 seed=12345 partind ;
    by churn;
    output out=out.telecomms_part_2 copyvars=(_all_);
run;

/***  Modify the Partition Indicator: 0 = Training & 1 = Validation  ****/
proc format;
    value _partind_ 0='Training' 1='Validation';
run;

/**Applying format to the variable churn**/
data out.telecomms_part_2;
    set out.telecomms_part_2;
    format _partind_ _partind_.;
run;

/**Prints the first 10 observations**/
proc print data=out.telecomms_part_2(obs=10);
    var _partind_;
run;

proc contents data=out.telecomms_part_2;
run;
/**************  Extra-Code with Added input variables *******************/
/*************************************************************************/
/***  Macro-Variable for Inputs for Model  ****/
%let VI_inputs=pymts_late_ltd ever_days_over_plan mou_total_pct_MOM
    curr_days_susp delinq_indicator times_delinq mou_roam_pct_MOM equip_age
    bill_data_usg_tot nbr_contracts_ltd rfm_score Est_HH_Income
    cs_med_home_value cs_pct_home_owner cs_ttl_pop cs_hispanic cs_caucasian
    cs_afr_amer cs_other cs_ttl_urban cs_ttl_rural cs_ttl_male cs_ttl_female
    cs_ttl_hhlds cs_ttl_mdage forecast_region mb_inclplan;

/*************************BUILDING THE PROC SUPERLEARNER MODEL ***************/

/*************************************************************************/
proc superlearner data=out.telecomms_part_2(where=(_PartInd_=1 )) seed=23456;
    target churn / level=nominal;
    input &VI_inputs / level=interval;
    baselearner 'GB' gradboost(minleafsize=20 maxdepth=10);
    baselearner 'Forest' forest(maxdepth=15 ntrees=100 inbagfraction=0.6
        minleafsize=5);
    baselearner 'DT' treesplit;
    baselearner 'Logistic' logselect(link=logit selection=stepwise);
    crossvalidation kfold=6;
    output out=ps.predict2_output2 copyvar=churn;
    store out=out.score_model_2;
run;

/******  Score Validation from Previous Model *******/
;

/***************************************************/
proc superlearner data=out.telecomms_part_2 restore=out.score_model_2;
    output out=out.score_model_2 learnerpred copyvars=(churn _partind_);
run;

proc print data=out.score_model_2 (obs=10);
run;

/**** ASSESSING EVENT CHURN by the Partition  ***/
;
ods output ROCInfo=out.roc_table_2;

/* <----- This is the ROC Curve table output */
proc assess data=out.score_model_2;
    var P_churnChurn;
    target churn / event="Churn" level=nominal;
    by _partind_ ;
    /** Output ROC table **/
run;
ods output close;

proc print data=out.roc_table_2;
run;

proc print data=out.roc_table_2;
run;

data out.roc_table_2;
    set out.roc_table_2;
    model='SuperLearner';
run;

/**** PROC SUPER ROC CURVE FOR _PARTIND_ FOR EVENT = 1 (TRAINING SET) *****/
;

/*************************************************************************/
proc sgplot data=out.roc_table_2;
    series x=FPR y=Sensitivity / lineattrs=(thickness=2 color=RED);
    by _partind_;
    lineparm x=0 y=0 slope=1 / transparency=0.5 lineattrs=(pattern=shortdash
        color=BLUE); /* diagonal */
    xaxis label="False Positive Rate (1 - Specificity)" min=0 max=1;
    yaxis label="True Positive Rate (Sensitivity)" min=0 max=1;
    title "ROC Curve PROC SUPERLEARNER";
run;

/***** ROC for Light-Gradient Boosting Model ****/
ods output ROCInfo=out.roc_table_LGB_2;

/* <----- This is the ROC Curve table output */
proc assess data=out.score_model_2;
    var LGB;
    target churn / event="Churn" level=nominal;
    by _partind_ ;
    /** Output ROC table **/
run;
ods output close;

proc print data=out.roc_table_LGB_2;
run;

proc sgplot data=out.roc_table_LGB_2;
    series x=FPR y=Sensitivity / lineattrs=(thickness=2 color=RED);
    by _partind_;
    lineparm x=0 y=0 slope=1 / transparency=0.5 lineattrs=(pattern=shortdash
        color=BLUE); /* diagonal */
    xaxis label="False Positive Rate (1 - Specificity)" min=0 max=1;
    yaxis label="True Positive Rate (Sensitivity)" min=0 max=1;
    title "Light Gradient Booosting BASELEARNER: ROC Curve";
run;

/**************    ROC Curve for Forest Model    ********************/
/***************    ROC Curve Information   *************************/
ods output ROCInfo=out.roc_table_Forest_2;

/* <----- This is the ROC Curve table output */
proc assess data=out.score_model_2;
    var Forest;
    target churn / event="Churn" level=nominal;
    by _partind_ ;
    /** Output ROC table **/
run;
ods output close;

proc print data=out.roc_table_Forest_2;
run;

proc sgplot data=out.roc_table_Forest_2;
    series x=FPR y=Sensitivity / lineattrs=(thickness=2 color=RED);
    by _partind_;
    lineparm x=0 y=0 slope=1 / transparency=0.5 lineattrs=(pattern=shortdash
        color=BLUE); /* diagonal */
    xaxis label="False Positive Rate (1 - Specificity)" min=0 max=1;
    yaxis label="True Positive Rate (Sensitivity)" min=0 max=1;
    title "Forest BASELEARNER: ROC Curve";
run;

/**************    ROC Curve for Decision Tree Model    ********************/
/***************    ROC Curve Information   *************************/
ods output ROCInfo=out.roc_table_DT_2;

/* <----- This is the ROC Curve table output */
proc assess data=out.score_model_2;
    var DT;
    target churn / event="Churn" level=nominal;
    by _partind_ ;
    /** Output ROC table **/
run;
ods output close;

proc print data=out.roc_table_DT_2;
run;

proc sgplot data=out.roc_table_DT_2;
    series x=FPR y=Sensitivity / lineattrs=(thickness=2 color=RED);
    by _partind_;
    lineparm x=0 y=0 slope=1 / transparency=0.5 lineattrs=(pattern=shortdash
        color=BLUE); /* diagonal */
    xaxis label="False Positive Rate (1 - Specificity)" min=0 max=1;
    yaxis label="True Positive Rate (Sensitivity)" min=0 max=1;
    title "Decision Tree BASELEARNER: ROC Curve";
run;

/***ROC CURVE for Logistic Regression Model**/
/******************    ROC Curve Information   *************************/
ods output ROCInfo=out.roc_table_Logistic_2;

/* <----- This is the ROC Curve table output */
proc assess data=out.score_model_2;
    var Logistic;
    target churn / event="Churn" level=nominal;
    by _partind_ ;

run;
ods output close;

proc print data=out.roc_table_Logistic_2;
run;

proc sgplot data=out.roc_table_Logistic_2;
    series x=FPR y=Sensitivity / lineattrs=(thickness=2 color=RED);
    by _partind_;
    lineparm x=0 y=0 slope=1 / transparency=0.5 lineattrs=(pattern=shortdash
        color=BLUE); /* diagonal */
    xaxis label="False Positive Rate (1 - Specificity)" min=0 max=1;
    yaxis label="True Positive Rate (Sensitivity)" min=0 max=1;
    title "Logistic Regression BASELEARNER: ROC Curve";
run;

/*********************************************************************************/

/************  Create a Data step that combines all ROC tables together **************/
data out.combined_roc_data_2;
    set out.roc_table_2 out.roc_table_Forest_2 out.roc_table_LGB_2
        out.roc_table_DT_2 out.roc_table_Logistic_2;
    if Variable='Forest' then model='Forest';
    else if Variable='LGB' then model='LGB';
    else if Variable='DT' then model='DT';
    else if Variable='Logistic' then model='Logistic';
    else if Variable='Superlearner' then model='Superlearner';
run;

proc sgplot data=out.combined_roc_data_2(where=(_partind_=1));
    series x=FPR y=Sensitivity / group=model;
    /*lineattrs=(thickness=2 color=RED);*/
    lineparm x=0 y=0 slope=1 / transparency=0.5 lineattrs=(pattern=shortdash
        color=BLUE); /* diagonal */
    xaxis label="False Positive Rate (1 - Specificity)" min=0 max=1;
    yaxis label="True Positive Rate (Sensitivity)" min=0 max=1;
    title "ROC Curve: PROC SUPERLEARNER vs BASELEARNERS";
run;

/** Optional Code: More inputs  **/
/*proc superlearner data=out.telecomms_part(where=partind=1);
target churn / level=interval;
input &num_inputs / level=interval;
baselearner 'LGB' lightgradboost;
baselearner 'Forest' forest;
output out=ps.predict_output copyvar=churn;
run;

%let num_inputs=lifetime_value avg_arpu_3m acct_age billing_cycle
nbr_contracts_ltd rfm_score Est_HH_Income cs_med_home_value
cs_pct_home_owner cs_ttl_pop cs_hispanic cs_caucasian cs_afr_amer cs_other
cs_ttl_urban cs_ttl_rural cs_ttl_male cs_ttl_female cs_ttl_hhlds
cs_ttl_mdage forecast_region mb_inclplan ever_days_over_plan
ever_times_over_plan data_device_age equip_age mfg_apple mfg_samsung mfg_htc
mfg_motorola mfg_lg mfg_nokia delinq_indicator times_delinq
count_of_suspensions_6m avg_days_susp calls_total calls_in_pk calls_in_offpk
calls_out_offpk calls_out_pk mou_total_pct_MOM mou_onnet_pct_MOM
mou_roam_pct_MOM voice_tot_bill_mou_curr tot_voice_chrgs_curr tot_drpd_pr1
bill_data_usg_m03 bill_data_usg_m06 bill_data_usg_m09 mb_data_usg_m01
mb_data_usg_m02 mb_data_usg_m03 mb_data_ndist_mo6m mb_data_usg_roamm01
mb_data_usg_roamm02 mb_data_usg_roamm03 tot_mb_data_curr
tot_mb_data_roam_curr bill_data_usg_tot tot_overage_chgs
data_prem_chrgs_curr nbr_data_cdrs avg_data_chrgs_3m avg_data_prem_chrgs_3m
avg_overage_chrgs_3m nbr_contacts calls_TS_acct open_tsupcomplnts
num_tsupcomplnts unsolv_tsupcomplnt wrk_orders days_openwrkorders
resolved_complnts calls_care_acct calls_care_3mavg_acct
calls_care_6mavg_acct res_calls_3mavg_acct res_calls_6mavg_acct
last_rep_sat_score network_mention service_mention price_mention times_susp
curr_days_susp pymts_late_ltd calls_care_ltd seconds_of_data_norm
seconds_of_data_log LOG_MB_Data_Usg_M04 LOG_MB_Data_Usg_M05
LOG_MB_Data_Usg_M06 LOG_MB_Data_Usg_M07 LOG_MB_Data_Usg_M08
LOG_MB_Data_Usg_M09;

/********************* Variable Importance ********************
proc hpsplit data=ps.telecomms maxdepth=50;
class churn;
model churn=_all_;
run;*/
***/
;
