from SVM.Resampling_SVM import ResampleSVM
from DisplayTernaryPhaseDiag import Display



display = Display()
SMV = ResampleSVM(display.FormatDataFromCSV("PD_P1_SVM/Experimental Data P1/ER2_T300.csv")[1])
display.DisplayTernaryPhaseScatter(SMV.generate_all_phase(SMV.data_in, 0.66, 18400, 0.001), 4, "Generated SVM")



