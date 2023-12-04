# Sampling

dmosopt implements various sampling strategies listed below:

<ul>
    <li v-for="i in ['glp', 'slh', 'lh', 'mc', 'sobol']">
        {{ i }} - <a href="https://github.com/iraikov/dmosopt/blob/main/dmosopt/sampling.py" target="_blank">
            {{ i }}()
        </a>
    </li>
</ul>

You may also point to your custom implementations by specifying a Python import path.