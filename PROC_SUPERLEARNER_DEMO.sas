/****************   LIBNAME STATEMENT     ***************/;

*libname ps '/home/student/Courses/PROC_SUPERLEARNER/data';
libname sl '/home/student/Courses/PROC_SUPERLEARNER/output';

libname mycas cas caslib="casuser";

/********************************************************/


data mycas.telecomms_data;
    set sl.tele_comms_clean;
run;


proc print data=mycas.telecomms_data (obs=10);
run;
/************  Partition Data to 50/50 Split  ********/

/*****************************************************/
proc partition data=mycas.telecomms_data samppct=70 seed=12345 partind;
    by churn;
    output out=mycas.telecomms_part copyvars=(_all_);
run;

proc print data=mycas.telecomms_part(obs=10);
    var churn _PartInd_;
run;

/***********    Training & Validation Split ***************/

proc freq data=mycas.telecomms_part;
    tables _partind_;
run;

/******************* PROC SUPERLEARNER MODEL *************************/;
%let VI_inputs=pymts_late_ltd ever_days_over_plan mou_total_pct_MOM
    curr_days_susp delinq_indicator times_delinq mou_roam_pct_MOM equip_age
    bill_data_usg_tot;

/********************************************************************/

proc superlearner data=mycas.telecomms_part(where=(_PartInd_= 1 )) seed=23456;
    target churn / level=nominal;
    input &VI_inputs / level=interval;
   * baselearner 'LGB' lightgradboost(leafsize=15 numbin=100);
    baselearner 'GB' gradboost;
    baselearner 'Forest' forest;
    baselearner 'Logistic' logselect;
    crossvalidation kfold=6;
    output out=mycas.predict_output copyvar=churn;
    store out=mycas.score_model;
run;

/******  Score Validation from Previous Model *******/;

/***************************************************/

proc superlearner data=mycas.telecomms_part restore=mycas.score_model;
    output out=mycas.score_data learnerpred copyvars=(churn _partind_);
run;

proc print data=mycas.score_data(obs=10);
run;

/**** ASSESSING EVENT CHURN by the Partition  ***/;

/********************  ROC Curve Information  ***************************/

ods output ROCInfo=mycas.roc_table;  /*<----- This is the ROC Curve table output */
proc assess data=mycas.score_data;
    var P_churnChurn;
    target churn / event="Churn" level=nominal;
    by _partind_ ;
    /** Output ROC table **/
run;
ods output close;

data mycas.roc_table;
    set mycas.roc_table;
    model='SuperLearner';
run;

/**** PROC SUPER ROC CURVE FOR _PARTIND_ FOR EVENT = 1 (TRAINING SET) ****/;


/*************************************************************************/


proc sgplot data=mycas.roc_table;
    series x=FPR y=Sensitivity / lineattrs=(thickness=2 color=RED);
    by _partind_;
    lineparm x=0 y=0 slope=1 / transparency=0.5 lineattrs=(pattern=shortdash
        color=BLUE); /* diagonal */
    xaxis label="False Positive Rate (1 - Specificity)" min=0 max=1;
    yaxis label="True Positive Rate (Sensitivity)" min=0 max=1;
    title "PROC SUPERLEARNER: ROC Curve";
run;

/********    ROC PROC SUPER VS BASELEARNERS   *************/;

/********************************************************/



/***** ROC for Gradient Boosting Model ****/
ods output ROCInfo=mycas.roc_table_GB;

/* <----- This is the ROC Curve table output */
proc assess data=mycas.score_data;
    var GB;
    target churn / event="Churn" level=nominal;
    by _partind_ ;
    /** Output ROC table **/
run;
ods output close;

proc print data=mycas.roc_table_GB;
run;

proc sgplot data=mycas.roc_table_GB;
    series x=FPR y=Sensitivity / lineattrs=(thickness=2 color=RED);
    by _partind_;
    lineparm x=0 y=0 slope=1 / transparency=0.5 lineattrs=(pattern=shortdash
        color=BLUE); /* diagonal */
    xaxis label="False Positive Rate (1 - Specificity)" min=0 max=1;
    yaxis label="True Positive Rate (Sensitivity)" min=0 max=1;
    title "Gradient Booosting BASELEARNER: ROC Curve";
run;


/**************    ROC Curve for Forest Model    ********************/

/***************    ROC Curve Information   *************************/
ods output ROCInfo=mycas.roc_table_Forest;

/* <----- This is the ROC Curve table output */
proc assess data=mycas.score_data;
    var Forest;
    target churn / event="Churn" level=nominal;
    by _partind_ ;
    /** Output ROC table **/
run;
ods output close;

proc print data=mycas.roc_table_Forest;
run;

proc sgplot data=mycas.roc_table_Forest;
    series x=FPR y=Sensitivity / lineattrs=(thickness=2 color=RED);
    by _partind_;
    lineparm x=0 y=0 slope=1 / transparency=0.5 lineattrs=(pattern=shortdash
        color=BLUE); /* diagonal */
    xaxis label="False Positive Rate (1 - Specificity)" min=0 max=1;
    yaxis label="True Positive Rate (Sensitivity)" min=0 max=1;
    title "Forest BASELEARNER: ROC Curve";
run;



/***ROC CURVE for Logistic Regression Model**/


/******************    ROC Curve Information   *************************/
ods output ROCInfo=mycas.roc_table_Logistic;

/* <----- This is the ROC Curve table output */
proc assess data=mycas.score_data;
    var Logistic;
    target churn / event="Churn" level=nominal;
    by _partind_ ;

run;
ods output close;

proc print data=mycas.roc_table_Logistic;
run;

proc sgplot data=mycas.roc_table_Logistic;
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
data mycas.combined_roc_data;
    set mycas.roc_table mycas.roc_table_forest mycas.roc_table_GB mycas.roc_table_Logistic;
    if Variable='Forest' then model='Forest';
    else if Variable='GB' then model='GB';
    else if Variable='Logistic' then model='Logistic';
    else if Variable='Superlearner' then model='Superlearner';
run;

proc sgplot data=mycas.combined_roc_data(where=(_partind_=1));
    series x=FPR y=Sensitivity / group=model;
        /*lineattrs=(thickness=2 color=RED);*/
    lineparm x=0 y=0 slope=1 / transparency=0.5 lineattrs=(pattern=shortdash
        color=BLUE); /* diagonal */
    xaxis label="False Positive Rate (1 - Specificity)" min=0 max=1;
    yaxis label="True Positive Rate (Sensitivity)" min=0 max=1;
    title "ROC Curve: PROC SUPERLEARNER vs BASELEARNERS";
run;
