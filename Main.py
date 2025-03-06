import wx
import subprocess
import shutil
import glob
import time
import os
import threading
import platform
#from win32com.shell import shell, shellcon

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
                from win32com.shell import shell, shellcon
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
        
        default_font = wx.Font(10, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, faceName="Segoe UI")
        self.SetFont(default_font)

        # Main vertical sizer
        vbox = wx.BoxSizer(wx.VERTICAL)

        # Read Type Choice
        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        hbox1.Add(wx.StaticText(panel, label="Select read type:"), flag=wx.RIGHT, border=8)
        self.read_type = wx.Choice(panel, choices=["Single-ended", "Paired-ended"])
        self.read_type.Bind(wx.EVT_CHOICE, self.on_read_type_change)
        hbox1.Add(self.read_type)
        vbox.Add(hbox1, flag=wx.LEFT | wx.TOP, border=10)

        # Path inputs for Read 1 and Read 2
        hbox2 = wx.BoxSizer(wx.HORIZONTAL)
        hbox2.Add(wx.StaticText(panel, label="Path to Read1:"), flag=wx.RIGHT, border=8)
        self.read1_path = wx.TextCtrl(panel, size=(300, -1))
        browse_read1 = wx.Button(panel, label="Browse")
        browse_read1.Bind(wx.EVT_BUTTON, self.on_browse_read1)
        hbox2.Add(self.read1_path, flag=wx.RIGHT, border=8)
        hbox2.Add(browse_read1)
        vbox.Add(hbox2, flag=wx.LEFT | wx.TOP, border=10)

        hbox3 = wx.BoxSizer(wx.HORIZONTAL)
        hbox3.Add(wx.StaticText(panel, label="Path to Read2:"), flag=wx.RIGHT, border=8)
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
        hbox4.Add(self.annotate_checkbox)
        hbox4.Add(wx.StaticText(panel, label="Quality to filter using FASTP:"), flag=wx.LEFT, border=40)
        self.qual = wx.TextCtrl(panel, size=(50, -1))
        hbox4.Add(self.qual, flag=wx.LEFT, border=5)
        vbox.Add(hbox4, flag=wx.LEFT | wx.TOP, border=10)

        # Species choice
        hbox5 = wx.BoxSizer(wx.HORIZONTAL)
        hbox5.Add(wx.StaticText(panel, label="Species:"), flag=wx.RIGHT, border=8)
        self.species = wx.Choice(panel, choices=["Escherichia coli", "Custom"])
        self.species.Bind(wx.EVT_CHOICE, self.on_species_change)
        hbox5.Add(self.species)
        vbox.Add(hbox5, flag=wx.LEFT | wx.TOP, border=10)

        # Model, PCA, Label, and Reference genome paths with Browse buttons
        def add_file_input(label, disable=True):
            hbox = wx.BoxSizer(wx.HORIZONTAL)
            hbox.Add(wx.StaticText(panel, label=label), flag=wx.RIGHT, border=8)
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

        # Run Button
        hbox6 = wx.BoxSizer(wx.HORIZONTAL)
        self.run_btn = wx.Button(panel, label="Predict")
        self.run_btn.Bind(wx.EVT_BUTTON, self.on_run_button)
        hbox6.Add(self.run_btn)
        vbox.Add(hbox6, flag=wx.LEFT | wx.TOP, border=10)

        # Output text area
        self.output_area = wx.TextCtrl(panel, size=(550, 200), style=wx.TE_MULTILINE | wx.TE_READONLY)
        vbox.Add(self.output_area, flag=wx.LEFT | wx.TOP | wx.EXPAND, border=10)

        panel.SetSizer(vbox)
        wx.CallAfter(self.set_licensing_info)

        # Window settings
        self.SetTitle("UnnamedApp")
        self.SetSize((635, 675))
        self.Centre()

    # Event handlers and pipeline code follow as in your initial code
    def set_licensing_info(self):
        self.output_area.SetValue("This software is licensed under the GNU Affero General Public License v3 (AGPL v3).\n"
        "Dependencies and their respective licenses:\n"
        "- Python: Python Software Foundation License\n"
        "- BWA: GPL License\n"
        "- Fastp: MIT License\n"
        "- SAMtools: MIT License\n"
        "- BCFtools: MIT License\n"
        "- VCFtools: GPL License\n"
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
        # Fetch and validate input
        end_type = "P" if self.read_type.GetStringSelection() == "Paired-ended" else "S"
        self.read_1 = self.read1_path.GetValue().replace('\\', '/')
        self.read_2 = self.read2_path.GetValue().replace('\\', '/') if end_type == "P" else ""
        
        species_string = self.species.GetStringSelection()
        species_dict = {
            "Escherichia coli" : "ecoli",
            "Custom" : "custom"
        }
        species_selection = species_dict[species_string]
        
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
        annotation_mode = "--mode annotation" if self.annotate_checkbox.IsChecked() and species_selection != "custom" else ""
        
        quality_filter = '15' if self.qual.GetValue() == '' else self.qual.GetValue()

        command_run = (
            f"docker run --rm -v {self.doc}/Unnamed/main/:/workspace kani nextflow run /workspace/script/main.nf"
            f" --ref {reference_sequence}"
            f" --read1 {input_1} {second_read}"
            f" --model {model_file}"
            f" --pca {pca_file}"
            f" --bed {annotation_file}"
            f" --labels {labels_file}"
            f" {annotation_mode}"
            f" --outdir /workspace/output"
            f" --qual {quality_filter}"
        )

        # Run the command and track the time
        start_time = time.time()
        process = subprocess.run(command_run, shell=True, stderr=subprocess.PIPE, text=True)
        end_time = time.time()

        # Calculate elapsed time
        elapsed_time = end_time - start_time
        minutes, seconds = divmod(elapsed_time, 60)

        # Find the output file and read its contents
        out_txt_name = glob.glob(f"{output_directory}{input_1_name}*output.txt")
        output_data = ""
        if out_txt_name:
            out_text_directory = out_txt_name[0]
            with open(out_text_directory, "r") as output_prediction:
                output_data = output_prediction.read()
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

    # You may reuse the methods for browsing files, running the pipeline, etc.

def main():
    app = wx.App()
    ex = UnnamedApp(None)
    ex.Show()
    app.MainLoop()

if __name__ == '__main__':
    main()
