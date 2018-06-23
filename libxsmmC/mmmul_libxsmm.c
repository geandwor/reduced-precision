#include <libxsmm_source.h>

#include <libxsmm.h>

#include <stdlib.h>

#include <stdio.h>
/*generate a random number between lower and upper using libxsmm_rand_f64() function to generate*/
float random_generatorbw(float lower, float upper){
    float fl =(float)(lower+(upper-lower)*libxsmm_rand_f64());
    return fl;
}

/*truncate_mask_fp32_bfp16 changes a 32 floating point to 16 bit point with same bit 
 * for exponent.
 * What it does is zeroing the 16 less significant bits of the 32 bit' number in littel endian fasion
 * the result will be so called bfloat16 number.
 
*/

void truncate_mask_fp32_bfp16(float* in, float* out, unsigned int len) {
  unsigned int i = 0;

  /* truncate buffer to bfp16 */
  for ( i = 0; i < len; ++i ) {
    union libxsmm_bfloat16_hp t;

    t.f = in[i];
    t.i[0] = 0;
    out[i] = t.f;
  }
}


int main()
{
    const float alpha = 1.0, beta = 1.0;
    /*const int m = 2, n = 2, k =2;*/
    libxsmm_blasint m = 2, n = 2, k =2;
    int ite= 100;
    /*initialize c with all the element of 0.0*/
    /*float*/
    float a[m*k], b[k*n], a1[m*k],b1[k*n],c[m*n],c1[m*n];
    /*int c[m*n], c1[m*n];*/
    /*libxsmm_bfloat16 cbf16[m*n];*/
    
    float lowerlimit[] = {1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,0.1,1,10,100};
    float upperlimit[] = {1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,0.1,1,10,100,180};
    float lower, upper;
    int len = sizeof(lowerlimit)/sizeof(float) ;
    int i,j,l,q,r;
    
    for(j=0;j<m;j++){
        for(q=0;q<n;q++)
        c[j*m+q] = 0.0;
    }
    
    for(i=0; i<len; i++){
        lower = lowerlimit[i];
        upper = upperlimit[i];
        
        for(l=0; l<3;l++){
            printf("Lower limit is %.8f, upper limit is %.8f\n",lower,upper);
            /*fill random number into the matrix a and b with value between lower and upper*/
            printf("In matrix a\n");
            for(j=0; j<m;j++){
                for(q=0;q<k;q++){
                 a[j*m+q] = random_generatorbw(lower,upper);
                 printf("the [%d][%d] element is %.8f\n",j,q,a[j*m+q]);   
                }
                
            
            }
            
            truncate_mask_fp32_bfp16(a,a1,(unsigned int)(m*k));
            printf("In matrix b\n");
            for(j=0;j<k;j++){
                for(q=0;q<n;q++){
                 b[j*k+q] = random_generatorbw(lower,upper);
                 printf("the [%d][%d] element is %.8f\n",j,q,b[j*k+q]);   
                }
            }
            truncate_mask_fp32_bfp16(b,b1,(unsigned int)(k*n));
            
            /*calcuation matrix multiplication using libxsmm_gemm*/
            libxsmm_sgemm(NULL,NULL, &m, &n, &k,&alpha, a, NULL, b, NULL, &beta, c, NULL);
            
            libxsmm_sgemm(NULL,NULL, &m, &n, &k,&alpha, a1, NULL, b1, NULL, &beta, c1, NULL);

            /*print element in c after gemm*/
            /* for(j=0;j<m;j++){
                for(q=0;q<n;q++){
                   
                    printf("the [%d][%d] element is %.8f\n",j,q,c[j][q]); 
                }
            }
             */
            /*convert each element in the calculated matrix to libxsmm_bfloat16*/
            printf("In matrix c\n");
            /*calculate the mm multiplication manually*/
            /*
            for(j=0;j<m;j++){
                for(q=0;q<n;q++){
                    for(r=0;r<k;r++)
                        c[j][q] += a[j][r]*b[r][q];
                    printf("the [%d][%d] element is %.8f\n",j,q,c[j][q]);   
                }
            }
            */
            /*convert single precision to bfloating16*/
            printf("In matrix of bf16\n");
            for(j=0;j<m;j++){
                for(q=0;q<n;q++){
                    /*cbf16[j*m+q]=(libxsmm_bfloat16)c[j*m+q];*/
                    /*cbf16[j][q]=(float)cbf16[j][q];*/
                    /*'hu' stands for short unsigned*/
                    printf("the [%d][%d] element in float32 is %.8f\n",j,q,c[j*m+q]);   

                    printf("the [%d][%d] element in bfloat16 is %.8f\n",j,q,c1[j*m+q]); 
                }
            }
            }
        
    
    }
    
}