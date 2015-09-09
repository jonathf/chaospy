/* Body of convergence test, shared between hcubature.c and
   pcubature.c.  We use an #include file because the two routines use
   somewhat different data structures, and define macros ERR(j) and
   VAL(j) to get the error and value estimates, respectively, for
   integrand j. */
{
     unsigned j;
#    define SQR(x) ((x) * (x))
     switch (norm) {
	 case ERROR_INDIVIDUAL:
	      for (j = 0; j < fdim; ++j)
		   if (ERR(j) > reqAbsError && ERR(j) > fabs(VAL(j))*reqRelError)
			return 0;
	      return 1;
	      
	 case ERROR_PAIRED:
	      for (j = 0; j+1 < fdim; j += 2) {
		   double maxerr, serr, err, maxval, sval, val;
		   /* scale to avoid overflow/underflow */
		   maxerr = ERR(j) > ERR(j+1) ? ERR(j) : ERR(j+1);
		   maxval = VAL(j) > VAL(j+1) ? VAL(j) : VAL(j+1);
		   serr = maxerr > 0 ? 1/maxerr : 1;
		   sval = maxval > 0 ? 1/maxval : 1;
		   err = sqrt(SQR(ERR(j)*serr) + SQR(ERR(j+1)*serr)) * maxerr;
		   val = sqrt(SQR(VAL(j)*sval) + SQR(VAL(j+1)*sval)) * maxval;
		   if (err > reqAbsError && err > val*reqRelError)
			return 0;
	      }
	      if (j < fdim) /* fdim is odd, do last dimension individually */
		   if (ERR(j) > reqAbsError && ERR(j) > fabs(VAL(j))*reqRelError)
			return 0;
	      return 1;

	 case ERROR_L1: {
	      double err = 0, val = 0;
	      for (j = 0; j < fdim; ++j) {
		   err += ERR(j);
		   val += fabs(VAL(j));
	      }
	      return err <= reqAbsError || err <= val*reqRelError;
	 }

	 case ERROR_LINF: {
	      double err = 0, val = 0;
	      for (j = 0; j < fdim; ++j) {
		   double absval = fabs(VAL(j));
		   if (ERR(j) > err) err = ERR(j);
		   if (absval > val) val = absval;
	      }
	      return err <= reqAbsError || err <= val*reqRelError;
	 }

	 case ERROR_L2: {
	      double maxerr = 0, maxval = 0, serr, sval, err = 0, val = 0;
	      /* scale values by 1/max to avoid overflow/underflow */
	      for (j = 0; j < fdim; ++j) {
		   double absval = fabs(VAL(j));
		   if (ERR(j) > maxerr) maxerr = ERR(j);
		   if (absval > maxval) maxval = absval;
	      }
	      serr = maxerr > 0 ? 1/maxerr : 1;
	      sval = maxval > 0 ? 1/maxval : 1;
	      for (j = 0; j < fdim; ++j) {
		   err += SQR(ERR(j) * serr);
		   val += SQR(fabs(VAL(j)) * sval);
	      }
	      err = sqrt(err) * maxerr;
	      val = sqrt(val) * maxval;
	      return err <= reqAbsError || err <= val*reqRelError;
	 }
     }
     return 1; /* unreachable */
}
