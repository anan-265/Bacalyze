import wx
import subprocess
import shutil
import glob
import time
import os
import threading
import platform

class UnnamedApp(wx.Frame):
    def __init__(self, *args, **kw):
        frame_style = wx.DEFAULT_FRAME_STYLE & ~(wx.RESIZE_BORDER | wx.MAXIMIZE_BOX)
        super(UnnamedApp, self).__init__(*args, **kw, style=frame_style)
        
        self.InitUI()
        self.read_1 = ""
        self.read_2 = ""
        self.init_directories()
    
    def get_doc_folder(self):
        if platform.system() == "Windows":
            try:
                from win32com.shell import shell, shellcon # type: ignore
                return shell.SHGetFolderPath(0, shellcon.CSIDL_PERSONAL, None, 0)
            except ImportError:
                raise ImportError("win32com is not installed. Install it using 'pip install pywin32'.")
        else:
            return os.path.expanduser("~/Documents")

    def init_directories(self):
        # Define paths
        self.doc = self.get_doc_folder()
        self.input_directory = f"{self.doc}/Unnamed/main/input/"
        self.output_directory = f"{self.doc}/Unnamed/main/output/"
        self.custom_directory = f"{self.doc}/Unnamed/main/custom/"

        # Create directories if they don't exist
        os.makedirs(self.input_directory, exist_ok=True)
        os.makedirs(self.output_directory, exist_ok=True)
        os.makedirs(self.custom_directory, exist_ok=True)

    def cleanup_directories(self):
        # Delete all files in input, output, and custom directories
        for directory in [self.input_directory, self.custom_directory]:
            files = glob.glob(f"{directory}*")
            for f in files:
                try:
                    os.remove(f)
                except Exception as e:
                    print(f"Error deleting {f}: {e}")

    def InitUI(self):
        panel = wx.Panel(self)
        
        default_font = wx.Font(10, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, faceName="Google Sans")
        self.SetFont(default_font)

        # Main vertical sizer
        vbox = wx.BoxSizer(wx.VERTICAL)

        # Read Type Choice
        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        hbox1.Add(wx.StaticText(panel, label="Select read type:"), flag=wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, border=8)
        self.read_type = wx.Choice(panel, choices=["Single-ended", "Paired-ended"])
        self.read_type.Bind(wx.EVT_CHOICE, self.on_read_type_change)
        hbox1.Add(self.read_type, flag=wx.ALIGN_CENTER_VERTICAL)
        vbox.Add(hbox1, flag=wx.LEFT | wx.TOP, border=10)

        # Path inputs for Read 1 and Read 2
        hbox2 = wx.BoxSizer(wx.HORIZONTAL)
        hbox2.Add(wx.StaticText(panel, label="Path to Read1:"), flag=wx.ALIGN_CENTER_VERTICAL, border=8)
        self.read1_path = wx.TextCtrl(panel, size=(300, -1))
        browse_read1 = wx.Button(panel, label="Browse")
        browse_read1.Bind(wx.EVT_BUTTON, self.on_browse_read1)
        hbox2.Add(self.read1_path, flag=wx.RIGHT, border=8)
        hbox2.Add(browse_read1)
        vbox.Add(hbox2, flag=wx.LEFT | wx.TOP, border=10)

        hbox3 = wx.BoxSizer(wx.HORIZONTAL)
        hbox3.Add(wx.StaticText(panel, label="Path to Read2:"), flag=wx.ALIGN_CENTER_VERTICAL, border=8)
        self.read2_path = wx.TextCtrl(panel, size=(300, -1))
        browse_read2 = wx.Button(panel, label="Browse")
        browse_read2.Bind(wx.EVT_BUTTON, self.on_browse_read2)
        hbox3.Add(self.read2_path, flag=wx.RIGHT, border=8)
        hbox3.Add(browse_read2)
        vbox.Add(hbox3, flag=wx.LEFT | wx.TOP, border=10)

        # Initially disable Read 2 controls
        self.read2_path.Disable()
        browse_read2.Disable()
        self.browse_read2 = browse_read2

        # Annotate checkbox and quality filter
        hbox4 = wx.BoxSizer(wx.HORIZONTAL)
        self.annotate_checkbox = wx.CheckBox(panel, label="Do you want to annotate variants?")
        hbox4.Add(self.annotate_checkbox, wx.ALIGN_CENTER_VERTICAL)
        hbox4.Add(wx.StaticText(panel, label="Quality to filter using FASTP:"), flag=wx.ALIGN_CENTER_VERTICAL | wx.LEFT, border=40)
        self.qual = wx.TextCtrl(panel, size=(50, 23))
        hbox4.Add(self.qual, flag=wx.LEFT, border=5)
        vbox.Add(hbox4, flag=wx.LEFT | wx.TOP, border=10)

        # Species choice
        hbox5 = wx.BoxSizer(wx.HORIZONTAL)
        hbox5.Add(wx.StaticText(panel, label="Species:"), flag=wx.ALIGN_CENTER_VERTICAL, border=8)
        self.species = wx.Choice(panel, choices=["Escherichia coli", "Custom"])
        self.species.Bind(wx.EVT_CHOICE, self.on_species_change)
        hbox5.Add(self.species)
        vbox.Add(hbox5, flag=wx.LEFT | wx.TOP, border=10)

        # Model, PCA, Label, and Reference genome paths with Browse buttons
        def add_file_input(label, disable=True):
            hbox = wx.BoxSizer(wx.HORIZONTAL)
            hbox.Add(wx.StaticText(panel, label=label), flag=wx.ALIGN_CENTER_VERTICAL, border=8)
            text_ctrl = wx.TextCtrl(panel, size=(300, -1))
            browse_button = wx.Button(panel, label="Browse")
            hbox.Add(text_ctrl, flag=wx.RIGHT, border=8)
            hbox.Add(browse_button)
            if disable:
                text_ctrl.Disable()
                browse_button.Disable()
            vbox.Add(hbox, flag=wx.LEFT | wx.TOP, border=10)
            return text_ctrl, browse_button

        self.model_path, self.browse_model = add_file_input("Load your model:")
        self.browse_model.Bind(wx.EVT_BUTTON, self.on_browse_model)
        self.pca_path, self.browse_pca = add_file_input("Load your pca transformer:")
        self.browse_pca.Bind(wx.EVT_BUTTON, self.on_browse_pca)
        self.label_path, self.browse_labels = add_file_input("Load your labels:")
        self.browse_labels.Bind(wx.EVT_BUTTON, self.on_browse_labels)
        self.ref_path, self.browse_ref = add_file_input("Load your reference genome:")
        self.browse_ref.Bind(wx.EVT_BUTTON, self.on_browse_ref)
        
        hbox6 = wx.BoxSizer(wx.HORIZONTAL)
        self.predict_checkbox = wx.CheckBox(panel, label="Predict AMR")
        hbox6.Add(self.predict_checkbox, wx.ALIGN_CENTER_VERTICAL)
        self.SNP_checkbox = wx.CheckBox(panel, label="Identify SNPs")
        hbox6.Add(self.SNP_checkbox, wx.ALIGN_CENTER_VERTICAL)
        self.predict_ARG_checkbox = wx.CheckBox(panel, label="Predict ARGs")
        hbox6.Add(self.predict_ARG_checkbox, wx.ALIGN_CENTER_VERTICAL)
        self.bulk_paired_checkbox = wx.CheckBox(panel, label="Bulk process")
        hbox6.Add(self.bulk_paired_checkbox, wx.ALIGN_CENTER_VERTICAL)
        vbox.Add(hbox6, flag=wx.LEFT | wx.TOP, border=10)

        hbox7 = wx.BoxSizer(wx.HORIZONTAL)
        self.assemble_checkbox = wx.CheckBox(panel, label="Assemble")
        hbox7.Add(self.assemble_checkbox, wx.ALIGN_CENTER_VERTICAL)
        self.annotate_genome_checkbox = wx.CheckBox(panel, label="Annotate Genome")
        hbox7.Add(self.annotate_genome_checkbox, wx.ALIGN_CENTER_VERTICAL)
        self.predict_MGE_checkbox = wx.CheckBox(panel, label="Predict MGEs")
        hbox7.Add(self.predict_MGE_checkbox, wx.ALIGN_CENTER_VERTICAL)
        vbox.Add(hbox7, flag=wx.LEFT | wx.TOP, border=10)

        hbox8 = wx.BoxSizer(wx.HORIZONTAL)
        self.identify_MGE_association_checkbox = wx.CheckBox(panel, label="Identify MGE associations")
        hbox8.Add(self.identify_MGE_association_checkbox, wx.ALIGN_CENTER_VERTICAL)
        self.kmer_checkbox = wx.CheckBox(panel, label="Identify kmer frequencies")
        hbox8.Add(self.kmer_checkbox, wx.ALIGN_CENTER_VERTICAL)
        self.assembled_genome_checkbox = wx.CheckBox(panel, label="Assembled Genome")
        hbox8.Add(self.assembled_genome_checkbox, wx.ALIGN_CENTER_VERTICAL)
        self.assembled_genome_checkbox.Bind(wx.EVT_CHECKBOX, self.on_assembled_genome_change)
        vbox.Add(hbox8, flag=wx.LEFT | wx.TOP, border=10)
        
        hbox9 = wx.BoxSizer(wx.HORIZONTAL)
        self.super_checkbox = wx.CheckBox(panel, label="Super mode")
        hbox9.Add(self.super_checkbox, wx.ALIGN_CENTER_VERTICAL)
        self.super_checkbox.Bind(wx.EVT_CHECKBOX, self.on_super_mode_change)
        vbox.Add(hbox9, flag=wx.LEFT | wx.TOP, border=10)
        panel.Layout()

        # Run Button
        hbox10 = wx.BoxSizer(wx.HORIZONTAL)
        self.run_btn = wx.Button(panel, label="Run")
        self.run_btn.Bind(wx.EVT_BUTTON, self.on_run_button)
        hbox10.Add(self.run_btn)
        vbox.Add(hbox10, flag=wx.LEFT | wx.TOP, border=10)

        # Output text area
        self.output_area = wx.TextCtrl(panel, size=(550, 200), style=wx.TE_MULTILINE | wx.TE_READONLY)
        vbox.Add(self.output_area, flag=wx.LEFT | wx.TOP | wx.EXPAND, border=10)

        panel.SetSizer(vbox)
        wx.CallAfter(self.set_licensing_info)

        # Window settings
        self.SetTitle("UnnamedApp")
        self.SetSize((800, 800))
        self.Centre()
    
    def on_assembled_genome_change(self, event):
        is_checked = self.assembled_genome_checkbox.IsChecked()
        self.read2_path.Enable(not is_checked)
        self.browse_read2.Enable(not is_checked)
        self.read_type.Enable(not is_checked)
        #self.bulk_paired_checkbox.Enable(not is_checked)

    def on_super_mode_change(self, event):
        is_super_mode = self.super_checkbox.IsChecked()
        self.predict_checkbox.SetValue(is_super_mode)
        self.SNP_checkbox.SetValue(is_super_mode)
        self.predict_ARG_checkbox.SetValue(is_super_mode)
        self.assemble_checkbox.SetValue(is_super_mode)
        self.annotate_genome_checkbox.SetValue(is_super_mode)
        self.predict_MGE_checkbox.SetValue(is_super_mode)
        self.identify_MGE_association_checkbox.SetValue(is_super_mode)
        self.kmer_checkbox.SetValue(is_super_mode)
        self.predict_checkbox.Enable(not is_super_mode)
        self.SNP_checkbox.Enable(not is_super_mode)
        self.predict_ARG_checkbox.Enable(not is_super_mode)
        self.assemble_checkbox.Enable(not is_super_mode)
        self.annotate_genome_checkbox.Enable(not is_super_mode)
        self.predict_MGE_checkbox.Enable(not is_super_mode)
        self.identify_MGE_association_checkbox.Enable(not is_super_mode)
        self.kmer_checkbox.Enable(not is_super_mode)

    def set_licensing_info(self):
        self.output_area.SetValue("This software is licensed under the GNU Affero General Public License v3 (AGPL v3).\n"
        "Dependencies and their respective licenses:\n"
        "- Python: Python Software Foundation License\n"
        "- BWA: GNU GPL License v3\n"
        "- Fastp: MIT License\n"
        "- SAMtools: MIT License\n"
        "- BCFtools: MIT License\n"
        "- VCFtools: GNU Lesser GPL License v3\n"
        "- Nextflow: Apache License 2.0\n"
        "- pandas: BSD 3-Clause License\n"
        "- scikit-learn: BSD 3-Clause License\n"
        "Users are responsible for ensuring compliance with each dependency's license.\n")
    
    def on_read_type_change(self, event):
        # Enable or disable Read 2 controls based on the selected read type
        is_paired_end = self.read_type.GetStringSelection() == "Paired-ended"
        self.read2_path.Enable(is_paired_end)
        self.browse_read2.Enable(is_paired_end)
        
    def on_species_change(self, event):
        is_custom = self.species.GetStringSelection() == "Custom"
        self.model_path.Enable(is_custom)
        self.browse_model.Enable(is_custom)
        self.pca_path.Enable(is_custom)
        self.browse_pca.Enable(is_custom)
        self.label_path.Enable(is_custom)
        self.browse_labels.Enable(is_custom)
        self.ref_path.Enable(is_custom)
        self.browse_ref.Enable(is_custom)

    # Method to open file dialog for Read 1
    def on_browse_read1(self, event):
        with wx.FileDialog(self, "Choose Read1 file", wildcard="All files (*.*)|*.*", style=wx.FD_OPEN) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_OK:
                path = fileDialog.GetPath()
                self.read1_path.SetValue(path.replace('\\', '/'))

    # Method to open file dialog for Read 2
    def on_browse_read2(self, event):
        with wx.FileDialog(self, "Choose Read2 file", wildcard="All files (*.*)|*.*", style=wx.FD_OPEN) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_OK:
                path = fileDialog.GetPath()
                self.read2_path.SetValue(path.replace('\\', '/'))
    
    def on_browse_model(self,event):
        with wx.FileDialog(self,"Open Model", wildcard="Sklearn Pickle files (*.pkl)|*.pkl", style = wx.FD_OPEN) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_OK:
                path = fileDialog.GetPath()
                self.model_path.SetValue(path.replace('\\', '/'))
    
    def on_browse_pca(self,event):
        with wx.FileDialog(self,"Open PCA Transformer", wildcard="Sklearn Pickle files (*.pkl)|*.pkl", style = wx.FD_OPEN) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_OK:
                path = fileDialog.GetPath()
                self.pca_path.SetValue(path.replace('\\', '/'))
                
    def on_browse_labels(self,event):
        with wx.FileDialog(self,"Open Label lists", wildcard="Text Document files (*.txt)|*.txt", style = wx.FD_OPEN) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_OK:
                path = fileDialog.GetPath()
                self.label_path.SetValue(path.replace('\\', '/'))
                
    def on_browse_ref(self,event):
        with wx.FileDialog(self,"Open your reference genome", wildcard="Genome Sequence Files (*.*)|*.*", style = wx.FD_OPEN) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_OK:
                path = fileDialog.GetPath()
                self.ref_path.SetValue(path.replace('\\', '/'))

    # Method triggered when the Run button is clicked
    def on_run_button(self, event):
        # Disable the Run button to prevent multiple clicks
        self.run_btn.Disable()
        self.output_area.SetValue("Running pipeline...\n")
        
        # Start the pipeline in a new thread
        threading.Thread(target=self.run_pipeline).start()

    def run_pipeline(self):
        def check_dependencies():
            dependencies = ["nextflow", "bwa", "samtools", "vcftools", "bcftools", "python3", "spades", "mefinder", "resfinder"]
            missing_dependencies = []

            for dep in dependencies:
                if shutil.which(dep) is None:
                    missing_dependencies.append(dep)

            # Check for Python packages
                try:
                    import sklearn
                except ImportError:
                    missing_dependencies.append("sklearn")

                try:
                    import Bio
                except ImportError:
                    missing_dependencies.append("biopython")

            if missing_dependencies:
                missing_str = ", ".join(missing_dependencies)
            wx.CallAfter(self.output_area.AppendText, f"Missing dependencies: {missing_str}\nPlease install them by referring to their respective documentation.\n")
            wx.CallAfter(self.run_btn.Enable)  # Re-enable the Run button
            return False
            

        if shutil.which("docker") is None:
            if not check_dependencies():
                return
            else:
                wx.CallAfter(self.output_area.AppendText, "Docker is not installed. Please install Docker and build the docker image using the given Dockerfile to proceed.\n")
                wx.CallAfter(self.run_btn.Enable)  # Re-enable the Run button
            return
        # Get the values of all input fields
        end_type = "P" if self.read_type.GetStringSelection() == "Paired-ended" else "S"
        self.read_1 = self.read1_path.GetValue().replace('\\', '/')
        self.read_2 = self.read2_path.GetValue().replace('\\', '/') if end_type == "P" else ""
        
        species_string = self.species.GetStringSelection()
        species_dict = {
            "Escherichia coli" : "ecoli",
            "Custom" : "custom"
        }

            
        self.custom_model_path = self.model_path.GetValue().replace('\\','/') if species_string == "Custom" else ""
        self.custom_pca_path = self.pca_path.GetValue().replace('\\','/') if species_string == "Custom" else ""
        self.custom_label_path = self.label_path.GetValue().replace('\\','/') if species_string == "Custom" else ""
        self.custom_ref_path = self.ref_path.GetValue().replace('\\','/') if species_string == "Custom" else ""

        if not os.path.exists(self.read_1) or (end_type == "P" and not os.path.exists(self.read_2)):
            wx.CallAfter(self.output_area.AppendText, "Invalid path(s) for reads. Please verify the file paths.\n")
            wx.CallAfter(self.run_btn.Enable)  # Re-enable the Run button
            return

        # Set input and output directories
        input_directory = f"{self.doc}/Unnamed/main/input/"
        output_directory = f"{self.doc}/Unnamed/main/output/"
        custom_directory = f"{self.doc}/Unnamed/main/custom/"
        
        custom_model_name = os.path.basename(self.custom_model_path) if self.custom_model_path else ""
        custom_pca_name = os.path.basename(self.custom_pca_path) if self.custom_pca_path else ""
        custom_label_name = os.path.basename(self.custom_label_path) if self.custom_label_path else ""
        custom_ref_name = os.path.basename(self.custom_ref_path) if self.custom_label_path else ""
        
        shutil.copy(self.read_1, input_directory)
        if self.read_2:
            shutil.copy(self.read_2, input_directory)
        
        pipeline_mode = ""
        pipeline_assembled = "no"

        if self.predict_checkbox.IsChecked():
            pipeline_mode = "snp"
        elif self.SNP_checkbox.IsChecked():
            pipeline_mode = "snp-no-predict"
        elif self.predict_ARG_checkbox.IsChecked():
            pipeline_mode = "arg"
        elif self.assemble_checkbox.IsChecked():
            pipeline_mode = "assemble"
        elif self.annotate_genome_checkbox.IsChecked():
            pipeline_mode = "genome_annotation"
        elif self.predict_MGE_checkbox.IsChecked():
            pipeline_mode = "mge"
        elif self.identify_MGE_association_checkbox.IsChecked():
            pipeline_mode = "arg_mge"
        elif self.kmer_checkbox.IsChecked():
            pipeline_mode = "kmer"
        if self.assembled_genome_checkbox.IsChecked():
            pipeline_assembled = "yes"
        
        if pipeline_mode in ['snp','snp-no-predict','super']:
            species_selection = species_dict[species_string]
        else:
            species_selection = ""
        
        if species_selection == "custom":
            shutil.copy(self.custom_model_path,custom_directory)
            shutil.copy(self.custom_label_path,custom_directory)
            shutil.copy(self.custom_pca_path,custom_directory)
            shutil.copy(self.custom_ref_path,custom_directory)
            

        nextflow_input = "/workspace/input/"
        reference_sequence = f"/workspace/custom/{custom_ref_name}" if custom_ref_name else f"/workspace/predefined/reference_genomes/{species_selection}.fasta"
        model_file = f"/workspace/custom/{custom_model_name}" if custom_model_name else f"/workspace/predefined/models/{species_selection}.pkl"
        pca_file = f"/workspace/custom/{custom_pca_name}" if custom_pca_name else f"/workspace/predefined/generalisers/{species_selection}.pkl"
        labels_file = f"/workspace/custom/{custom_label_name}" if custom_label_name else f"/workspace/predefined/label_lists/{species_selection}.txt"
        annotation_file = f"/workspace/predefined/reference_annotations/{species_selection}.bed"
        input_1_name = os.path.basename(self.read_1)
        input_1 = f"{nextflow_input}{input_1_name}"
        input_2 = f"{nextflow_input}{os.path.basename(self.read_2)}" if self.read_2 else ""

        second_read = f"--read2 {input_2}" if input_2 else ""
        
        # Check if the annotation checkbox is checked
        annotation_mode = "--annotation yes" if self.annotate_checkbox.IsChecked() and species_selection != "custom" else ""
        
        quality_filter = '15' if self.qual.GetValue() == '' else self.qual.GetValue()
        
        reads_param = f" --read1 {input_1} {second_read}"

        if species_selection == "" or species_selection == "custom":
            self.arg_species = '--species other'
        else:
            self.arg_species = f'--species {species_selection}'

        command_run = (
            f"docker run -it -v {self.doc}/Unnamed/main/:/workspace kani nextflow run /workspace/script/main.nf"
            f" --ref {reference_sequence} {reads_param}"
            f" --model {model_file}"
            f" --pca {pca_file}"
            f" --bed {annotation_file}"
            f" --labels {labels_file}"
            f" {annotation_mode}"
            f" --mode {pipeline_mode}"
            f" --outdir /workspace/output/"
            f" --qual {quality_filter}"
            f" {self.arg_species}"
            f" --assembled {pipeline_assembled}"
        )

        # Run the command and track the time
        start_time = time.time()
        process = subprocess.run(command_run, shell=True, stderr=subprocess.PIPE, text=True)
        wx.CallAfter(self.output_area.AppendText, f"\nCommand executed:\n{command_run}\n")
        end_time = time.time()

        # Calculate elapsed time
        elapsed_time = end_time - start_time
        minutes, seconds = divmod(elapsed_time, 60)

        # Find the output file and read its contents
        out_txt_name = glob.glob(f"{output_directory}{input_1_name}*output.txt")
        output_data = ""
        if out_txt_name and pipeline_mode == "snp" and process.returncode == 0:
            out_text_directory = out_txt_name[0]
            with open(out_text_directory, "r") as output_prediction:
                output_data = output_prediction.read()
        elif pipeline_mode == "assemble" and process.returncode == 0:
            output_data = f"Reads Assembled successfully using SPAdes. Output files were saved in the {output_directory} directory."
        elif pipeline_mode == "genome_annotation" and process.returncode == 0:
            output_data = f"Genome Annotated successfully using Prokka. Output files were saved in the {output_directory} directory."
        elif pipeline_mode == "mge" and process.returncode == 0:
            output_data = f"MGEs Predicted successfully. Associated MGEs with genes were written as csv file in the {output_directory} directory."
        elif pipeline_mode == "arg_mge" and process.returncode == 0:
            output_data = f"MGEs and ARGs Identified successfully. Associated MGEs with genes and ARGs with MGEs were written as csv file in the {output_directory} directory."
        elif pipeline_mode == "kmer" and process.returncode == 0:
            output_data = f"Kmer frequencies Identified successfully. Kmer frequencies were written as csv file in the {output_directory} directory."
        elif pipeline_mode == "snp-no-predict" and process.returncode == 0:
            output_data = f"SNPs Identified successfully. Output files were saved in the {output_directory} directory."
        elif pipeline_mode == "arg" and process.returncode == 0:
            output_data = f"ARGs Predicted successfully using ResFinder. Output files were saved in the {output_directory} directory."
        else:
            output_data = "Output file not found. Please check the pipeline execution."

        # Update the GUI with the output
        wx.CallAfter(self.output_area.SetValue, output_data)
        wx.CallAfter(self.output_area.AppendText, f"\n\nCommand executed in {int(minutes)} minutes and {int(seconds)} seconds.\n")

        # Show any errors if the process failed
        if process.returncode != 0:
            wx.CallAfter(self.output_area.AppendText, f"\nError: {process.stderr}")

        # Re-enable the Run button after completion
        wx.CallAfter(self.cleanup_directories)
        wx.CallAfter(self.run_btn.Enable)


def main():
    app = wx.App()
    ex = UnnamedApp(None)
    ex.Show()
    app.MainLoop()

if __name__ == '__main__':
    main()
