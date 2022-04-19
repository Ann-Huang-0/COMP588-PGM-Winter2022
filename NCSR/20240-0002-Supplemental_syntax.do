/*-------------------------------------------------------------------------*
 |                                                                         
 |            STATA SUPPLEMENTAL SYNTAX FILE FOR ICPSR 20240
 |         COLLABORATIVE PSYCHIATRIC EPIDEMIOLOGY SURVEYS (CPES),
 |                       2001-2003 [UNITED STATES]
 |         (DATASET 0002: NATIONAL COMORBIDITY SURVEY REPLICATION
 |                          (NCS-R), 2001-2003)
 |
 | The program replaces user-defined numeric missing values (e.g., -9)
 | with extended system missing (e.g., '.r')  Note that Stata allows you
 | to specify up to 27 unique missing value codes.  Only variables with
 | user-defined missing values are included in this program.
 |
 | To apply the missing value recodes, users need to first open the
 | Stata data file on their system, apply the missing value recodes if
 | desired, then save a new copy of the data file with the missing values
 | applied.  Users are strongly advised to use a different filename when
 | saving the new file.
 |
 *------------------------------------------------------------------------*/

/***************************************************************************

 Extended Missing Values

 This section will recode numeric values (i.e., -8) with extended missing 
 values (i.e., '.d').  By default the code in this section is commented out.
 Users wishing to apply the extended missing values should remove the comment
 at the beginning and end of this section.   Note that Stata allows you to 
 specify up to 27 unique extended missing value codes.

****************************************************************/

foreach v of varlist _all {
   capture replace `v' = .n if `v' == -7
   capture replace `v' = .d if `v' == -8
   capture replace `v' = .r if `v' == -9
}
