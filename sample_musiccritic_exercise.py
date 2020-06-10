from . import ExerciseBase
import sample_assessment_piano as piano
from django.conf import settings

class HanonPianoExercise(ExerciseBase):

    def estimate_grades(self):
        # Necessary templates
        ly_path = self.exercise.data_files.get(file_type="LY").file.path
        json_path = self.exercise.data_files.get(file_type="JSON").file.path

        # Finding latency
        latency = self._get_latency(self.sound.submission)
        results = piano.single_file_assessment(self.sound.original_upload.path, ex_num, latency)
        self.chart_bytes = results["ImageBytes"]

        return results

    class Meta:
        display_name = "Hanon: Piano"
        provides_chart = True
        disable_echo_cancellation = True
        fields = ['score', 'description', 'countdown', 'backing', 'melody_length', 'bpm', 'lilypond_template', 'json_template']