import wx
import subprocess
import shutil
import glob
import time
import os
import threading

class UnnamedApp(wx.Frame):
    def __init__(self, *args, **kw):
        frame_style = wx.DEFAULT_FRAME_STYLE & ~(wx.RESIZE_BORDER | wx.MAXIMIZE_BOX)
        super(UnnamedApp, self).__init__(*args, **kw, style=frame_style)
        
        self.InitUI()
        self.read_1 = ""
        self.read_2 = ""
        
        # Display licensing information in the output area
        self.display_licensing_info()

    def InitUI(self):
        panel = wx.Panel(self)

        # Read Type Choice
        wx.StaticText(panel, label="Select read type:", pos=(20, 20))
        self.read_type = wx.Choice(panel, choices=["Single-ended", "Paired-ended"], pos=(150, 20))
        self.read_type.Bind(wx.EVT_CHOICE, self.on_read_type_change)

        # Path inputs for Read 1 and Read 2 with Browse Buttons
        wx.StaticText(panel, label="Path to Read1:", pos=(20, 60))
        self.read1_path = wx.TextCtrl(panel, pos=(150, 60), size=(300, -1))
        browse_read1 = wx.Button(panel, label="Browse", pos=(460, 60))
        browse_read1.Bind(wx.EVT_BUTTON, self.on_browse_read1)

        wx.StaticText(panel, label="Path to Read2:", pos=(20, 100))
        self.read2_path = wx.TextCtrl(panel, pos=(150, 100), size=(300, -1))
        browse_read2 = wx.Button(panel, label="Browse", pos=(460, 100))
        browse_read2.Bind(wx.EVT_BUTTON, self.on_browse_read2)

        # Initially disable Read 2 controls
        self.read2_path.Disable()
        browse_read2.Disable()

        self.browse_read2 = browse_read2

        # Other controls
        self.annotate_checkbox = wx.CheckBox(panel, label="Do you want to annotate variants?", pos=(20, 140))
        
        wx.StaticText(panel, label="Quality to filter using FASTP:", pos=(250, 140))
        self.qual = wx.TextCtrl(panel, pos=(405, 140), size=(20, -1))
        
        wx.StaticText(panel, label="Species:", pos=(20, 180))
        self.species = wx.Choice(panel, choices=["Escherichia coli", "Custom"], pos=(70, 180))
        self.species.Bind(wx.EVT_CHOICE, self.on_species_change)
        
        wx.StaticText(panel,label="Load your model:",pos = (20,220))
        self.model_path = wx.TextCtrl(panel, pos = (120,220), size = (50,-1))
        self.browse_model = wx.Button(panel, label="Browse", pos=(180, 220))
        self.browse_model.Bind(wx.EVT_BUTTON, self.on_browse_model)
        
        self.model_path.Disable()
        self.browse_model.Disable()
        
        wx.StaticText(panel,label="Load your pca transformer:",pos = (20,260))
        self.pca_path = wx.TextCtrl(panel, pos = (170,260), size = (50,-1))
        self.browse_pca = wx.Button(panel, label="Browse", pos=(230, 260))
        self.browse_pca.Bind(wx.EVT_BUTTON, self.on_browse_pca)
        
        self.pca_path.Disable()
        self.browse_pca.Disable()
        
        wx.StaticText(panel,label="Load your labels:",pos = (20,300))
        self.label_path = wx.TextCtrl(panel, pos = (125,300), size = (50,-1))
        self.browse_labels = wx.Button(panel, label="Browse", pos=(185, 300))
        self.browse_labels.Bind(wx.EVT_BUTTON, self.on_browse_labels)
        
        self.label_path.Disable()
        self.browse_labels.Disable()

        # Run Button
        self.run_btn = wx.Button(panel, label="Predict", pos=(20, 330))
        self.run_btn.Bind(wx.EVT_BUTTON, self.on_run_button)

        # Output text area
        self.output_area = wx.TextCtrl(panel, pos=(20, 380), size=(550, 200), style=wx.TE_MULTILINE | wx.TE_READONLY)

        # Window settings
        self.SetTitle("UnnamedApp")
        self.SetSize((635, 635))
        self.Centre()

    def display_licensing_info(self):
        licensing_info = (
            "Licensing Information:\n"
            "This software is licensed under the GNU Affero General Public License v3 (AGPL v3).\n"
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
            "Users are responsible for ensuring compliance with each dependency's license.\n"
        )

        # Display the licensing information in the output area
        wx.CallAfter(self.output_area.SetValue, licensing_info)

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

    # Method triggered when the Run button is clicked
    def on_run_button(self, event):
        # Disable the Run button to prevent multiple clicks
        self.run_btn.Disable()
        self.output_area.SetValue("Running pipeline...\n")
        
        # Start the pipeline in a new thread
        threading.Thread(target=self.run_pipeline).start()

    def run_pipeline(self):
        # (rest of the pipeline code)

def main():
    app = wx.App()
    ex = UnnamedApp(None)
    ex.Show()
    app.MainLoop()

if __name__ == '__main__':
    main()
